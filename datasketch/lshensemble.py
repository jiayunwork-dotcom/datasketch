import logging
import pickle
import struct
from collections import Counter, deque
from collections.abc import Generator, Hashable, Iterable
from numbers import Integral
from typing import NamedTuple, Optional

import numpy as np

from datasketch.lsh import MinHashLSH, integrate
from datasketch.lshensemble_partition import optimal_partitions
from datasketch.minhash import MinHash
from datasketch.storage import (
    UnsupportedStorageError,
    _random_name,
    supports_paged_bucket_scan,
)

logger = logging.getLogger(__name__)


class MergeNotSupportedError(NotImplementedError, ValueError):
    """Raised when an index cannot be merged in the requested way.

    This is both a :class:`NotImplementedError` and a :class:`ValueError` so
    callers that check for either convention can handle it. It is raised
    before the index (or the other index) is modified.
    """


class _RetainedEntry(NamedTuple):
    """Immutable copy of one indexed set, kept only by mergeable indexes.

    ``hashvalues`` is an immutable ``bytes`` snapshot in native uint64 byte
    order. It is never a view on a caller-owned array, so later changes to the
    caller's MinHash cannot alter the merge result.
    """

    key: Hashable
    hashvalues: bytes
    size: int


class _RetainedSketch:
    """A read-only MinHash-shaped view over retained hash value bytes.

    The LSH only needs ``len(sketch)`` and ``sketch.hashvalues[start:end]``.
    This view exposes those without copying a full key-count by num_perm
    matrix: ``np.frombuffer`` returns a read-only view of the retained bytes.
    """

    __slots__ = ("_hashvalues", "seed")

    def __init__(self, hashvalues: bytes, seed: int) -> None:
        self._hashvalues = hashvalues
        self.seed = seed

    @property
    def hashvalues(self) -> np.ndarray:
        return np.frombuffer(self._hashvalues, dtype=np.uint64)

    def __len__(self) -> int:
        return len(self._hashvalues) // 8


class _InvalidIntegerError(ValueError, TypeError):
    """Raised when a budget or set size is not a positive integer.

    It is both a :class:`ValueError` and a :class:`TypeError` because such an
    argument can be wrong in either way (non-integral or non-positive).
    """


def _validate_positive_int(value, name: str) -> int:
    """Validate a positive integer argument, returning it as a plain int."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise _InvalidIntegerError("%s must be a positive integer, got %r" % (name, value))
    value = int(value)
    if value <= 0:
        raise _InvalidIntegerError("%s must be a positive integer, got %d" % (name, value))
    return value


def _validate_budget(budget: int) -> int:
    """Validate a per-page budget, which must be a positive integer."""
    return _validate_positive_int(budget, "budget")


def _validate_size(size: int) -> int:
    """Validate a query set size, which must be a positive integer."""
    return _validate_positive_int(size, "size")


class MinHashLSHEnsemblePage(NamedTuple):
    """One page of candidates from a paged :class:`MinHashLSHEnsemble` query.

    Attributes:
        keys (list): The keys delivered by this page, in delivery order.
        budget_used (int): The number of budget units spent by this page. One
            unit is charged for each bucket lookup and for each key read,
            including reads of keys that were already delivered.
        done (bool): Whether the whole query has been read out, so no further
            page will deliver keys.

    """

    keys: list
    budget_used: int
    done: bool


class MinHashLSHEnsembleQuery:
    """A resumable, budgeted cursor over the candidates of one Ensemble query.

    Create it with :meth:`MinHashLSHEnsemble.query_paged` and read it page by
    page with :meth:`next_page`. The query's MinHash values and set size are
    snapshotted when the cursor is created, so later changes to the caller's
    MinHash do not affect the pages. Several cursors may be advanced
    interleaved, since each keeps its own read positions.

    Scheduling: a page's budget is spent in three phases, and every position
    is kept between pages.

    1. *Probe.* Band 0 of every partition the query touches is looked up, in
       partition order; a non-empty bucket found by a probe immediately yields
       one key, so the first page already shows candidates from every
       partition instead of only the first partition's hot bucket.
    2. *Finish lookups.* The remaining bands of every partition are looked up
       in partition-round-robin order (band 1 of each partition, then band 2,
       ...). A bucket found here also yields one key immediately.
    3. *Drain.* The parked live buckets are read round-robin, one key per turn.

    This ordering is what makes a small bucket in a late partition reachable
    early: lookups cost at most one unit per selected band, a newly found
    bucket delivers its first key at once, and the round-robin then reaches
    any parked bucket within as many reads as there are live buckets --
    independent of how many keys a hot bucket holds. Lookups are therefore
    never postponed behind reads from a hot bucket. The visible price is that
    a page can end in phase 2 with no new keys (every remaining bucket was
    already delivered), yet work remains, so `done` stays False. The retained
    state is one lazy read cursor per live bucket plus the set of delivered
    keys, so resuming never rescans or copies a bucket; memory grows with the
    number of colliding buckets, not with how many keys they hold.

    Args:
        ensemble (MinHashLSHEnsemble): The index to query.
        minhash (MinHash): The MinHash of the query set.
        size (int): The size (number of unique items) of the query set.

    Raises:
        ValueError: If the MinHash length does not match the index, if `size`
            is not a positive integer, or if the index does not use a
            dict-backed bucket storage that can be scanned lazily. The last
            check happens before any bucket is accessed.

    """

    def __init__(self, ensemble, minhash, size):
        self._ensemble = ensemble
        self._size = _validate_size(size)
        if len(minhash) != ensemble.h:
            raise ValueError("Expecting minhash with length %d, got %d" % (ensemble.h, len(minhash)))
        # Snapshot the query sketch: pages must be based on the MinHash as it
        # was when the query started.
        self._hashvalues = np.array(minhash.hashvalues, copy=True)

        # Reject an unsupported bucket backend now, before any bucket is
        # touched. The check is on the storage type of every partition, not on
        # the bands this query happens to select, so a query over an empty
        # partition cannot slip past it.
        for index in ensemble.indexes:
            for lsh in index.values():
                if not supports_paged_bucket_scan(lsh.hashtables[0]):
                    raise UnsupportedStorageError(
                        "Paged queries require bucket storage that supports lazy resumable reads, "
                        "got %s; use the default dict storage" % type(lsh.hashtables[0]).__name__
                    )

        # Select the same bands per partition as `query` does.
        lookups = []
        self._prepickle = False
        for i, index in enumerate(ensemble.indexes):
            upper = ensemble.uppers[i]
            if upper is None:
                continue
            b, r = ensemble._get_optimal_param(upper, self._size)
            lsh = index[r]
            b = int(b)
            self._prepickle = lsh.prepickle
            if b > 0:
                lookups.append((lsh, b))

        # Phase 1 probes band 0 of every partition; phase 2 finishes the
        # remaining bands in partition-round-robin order; phase 3 drains.
        self._probe_lookups = deque((lsh, 0, b) for lsh, b in lookups)
        self._pending_lookups = deque()
        # Each live bucket is [cursor, keys remaining]. The remaining count
        # comes from the lookup, so the bucket is dropped the moment its last
        # key is read and `done` is exact on the page that delivers it.
        self._cursors = deque()
        self._delivered = set()
        self._closed = False
        self._done = not self._probe_lookups and not self._pending_lookups
        self._budget_used = 0

    @property
    def done(self) -> bool:
        """bool: Whether the whole query has been read out."""
        return self._done

    @property
    def closed(self) -> bool:
        """bool: Whether this cursor has been closed."""
        return self._closed

    @property
    def delivered_count(self) -> int:
        """int: The number of distinct keys delivered so far."""
        return len(self._delivered)

    @property
    def budget_used(self) -> int:
        """int: The total number of budget units spent by all pages so far."""
        return self._budget_used

    def _has_work(self) -> bool:
        return bool(self._probe_lookups or self._pending_lookups or self._cursors)

    def _read_from(self, live):
        """Read one key from a specific live bucket.

        Args:
            live (list): ``[cursor, keys remaining]``.

        Returns:
            tuple: ``(has_key, key)``. A live bucket whose cursor has more
            keys is put back into the round-robin.

        """
        cursor, remaining = live
        try:
            key = next(cursor)
        except StopIteration:  # pragma: no cover - storage shrank mid-query
            return False, None
        if remaining > 1:
            live[1] = remaining - 1
            self._cursors.append(live)
        if self._prepickle:
            key = pickle.loads(key)
        return True, key

    def _read_one(self):
        """Read one key from the round-robin of live buckets.

        Returns:
            tuple: ``(has_key, key)``. ``has_key`` is False only when no live
            bucket is left. Exhausted buckets are dropped without being
            charged, which can only happen if the storage shrank mid-query.

        """
        while self._cursors:
            has_key, key = self._read_from(self._cursors.popleft())
            if has_key:
                return True, key
        return False, None

    def _lookup(self, pending):
        """Perform one bucket lookup and register any non-empty bucket.

        Args:
            pending (tuple): ``(lsh, band, band_count)`` for this lookup. If
                the partition has more bands, the next one is queued for the
                finish-lookups phase.

        Returns:
            list | None: A newly created live bucket ``[cursor, keys remaining]``
            if the bucket was non-empty, otherwise None. The caller decides
            whether to read it immediately or to park it in the round-robin.

        """
        lsh, band, band_count = pending
        bucket_key, bucket_len = lsh._paged_bucket_lookup(self._hashvalues, band)
        live = None
        if bucket_len:
            live = [lsh._paged_bucket_cursor(band, bucket_key), bucket_len]
        if band + 1 < band_count:
            self._pending_lookups.append((lsh, band + 1, band_count))
        return live

    def next_page(self, budget: int) -> MinHashLSHEnsemblePage:
        """Read the next page of candidates, spending at most `budget` units.

        One unit is charged for each bucket lookup (including buckets that are
        empty or that were already hit by another band) and for each key read
        out of a bucket (including keys that were already delivered). The page
        stops as soon as the budget is exhausted, keeping every read position
        for the next call.

        Args:
            budget (int): The positive number of units this page may spend.

        Returns:
            MinHashLSHEnsemblePage: The delivered keys, the units spent, and
            whether the query is exhausted.

        Raises:
            RuntimeError: If the cursor has been closed.
            ValueError: If `budget` is not a positive integer. The cursor's
                position is unchanged, so a corrected budget can be used.

        Note:
            A page may deliver no new keys while the query is not done, for
            example when its budget paid only for bucket lookups or when every
            key found was already delivered by another band. Keep calling
            :meth:`next_page` until `done` is True.

        """
        if self._closed:
            raise RuntimeError("This paged query is closed")
        budget = _validate_budget(budget)
        spent = 0
        page_keys = []

        def read_fresh(live):
            """Read one key from a bucket just found by a lookup."""
            nonlocal spent
            if spent >= budget:
                self._cursors.append(live)
                return
            has_key, key = self._read_from(live)
            if has_key:
                spent += 1
                if key not in self._delivered:
                    self._delivered.add(key)
                    page_keys.append(key)

        while spent < budget:
            # Phase 1: probe band 0 of the next partition. A bucket found by
            # the probe is read immediately, so the page shows candidates from
            # every partition early instead of only from the first hot bucket.
            if self._probe_lookups:
                live = self._lookup(self._probe_lookups.popleft())
                spent += 1
                if live is not None:
                    read_fresh(live)
                continue

            # Phase 2: finish the remaining bands. Lookups are not postponed
            # behind reads from a hot bucket, so a small bucket in a late band
            # is found after at most one lookup per selected band. Reading the
            # freshly found bucket right away means the key it holds is
            # delivered even when other buckets are full; the round-robin only
            # competes for later reads.
            if self._pending_lookups:
                live = self._lookup(self._pending_lookups.popleft())
                spent += 1
                if live is not None:
                    read_fresh(live)
                continue

            # Phase 3: drain the live buckets round-robin, one key per turn.
            if not self._cursors:
                break
            has_key, key = self._read_one()
            if not has_key:
                break
            spent += 1
            if key not in self._delivered:
                self._delivered.add(key)
                page_keys.append(key)

        self._budget_used += spent
        if not self._has_work():
            self._done = True
        return MinHashLSHEnsemblePage(page_keys, spent, self._done)

    # ``page`` reads naturally for callers that treat the cursor as a series
    # of pages; it is exactly ``next_page``.
    page = next_page

    def pages(self, budget: int) -> Generator[MinHashLSHEnsemblePage, None, None]:
        """Yield pages with the same per-page `budget` until the query is done.

        Args:
            budget (int): The positive number of units each page may spend.

        Yields:
            MinHashLSHEnsemblePage: The pages, ending with the page whose
            ``done`` is True.

        """
        budget = _validate_budget(budget)

        def generator():
            while not self._closed:
                page = self.next_page(budget)
                yield page
                if page.done:
                    return

        return generator()

    def close(self) -> None:
        """Close the cursor and release its read positions.

        After closing, :meth:`next_page` raises :class:`RuntimeError`. Closing
        an already closed cursor is a no-op.
        """
        self._closed = True
        self._probe_lookups.clear()
        self._pending_lookups.clear()
        for live in self._cursors:
            close = getattr(live[0], "close", None)
            if close is not None:
                close()
        self._cursors.clear()

    def __enter__(self) -> "MinHashLSHEnsembleQuery":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __repr__(self):
        return "MinHashLSHEnsembleQuery(done=%r, closed=%r, delivered=%d)" % (
            self._done,
            self._closed,
            len(self._delivered),
        )


def _false_positive_probability(threshold, b, r, xq):
    """Compute the false positive probability given the containment threshold.
    xq is the ratio of x/q.
    """
    _probability = lambda t: 1 - (1 - (t / (1 + xq - t)) ** float(r)) ** float(b)
    if xq >= threshold:
        a, _err = integrate(_probability, 0.0, threshold)
        return a
    a, _err = integrate(_probability, 0.0, xq)
    return a


def _false_negative_probability(threshold, b, r, xq):
    """Compute the false negative probability given the containment threshold."""
    _probability = lambda t: 1 - (1 - (1 - (t / (1 + xq - t)) ** float(r)) ** float(b))
    if xq >= 1.0:
        a, _err = integrate(_probability, threshold, 1.0)
        return a
    if xq >= threshold:
        a, _err = integrate(_probability, threshold, xq)
        return a
    return 0.0


def _optimal_param(threshold, num_perm, max_r, xq, false_positive_weight, false_negative_weight):
    """Compute the optimal parameters that minimizes the weighted sum
    of probabilities of false positive and false negative.
    xq is the ratio of x/q.
    """
    min_error = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        for r in range(1, max_r + 1):
            if b * r > num_perm:
                continue
            fp = _false_positive_probability(threshold, b, r, xq)
            fn = _false_negative_probability(threshold, b, r, xq)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt


class MinHashLSHEnsemble:
    """The :ref:`minhash_lsh_ensemble` index. It supports
    :ref:`containment` queries.
    The implementation is based on
    `E. Zhu et al. <http://www.vldb.org/pvldb/vol9/p1185-zhu.pdf>`_.

    Args:
        threshold (float): The Containment threshold between 0.0 and
            1.0. The initialized LSH Ensemble will be optimized for the threshold by
            minizing the false positive and false negative.
        num_perm (int): The number of permutation functions used
            by the MinHash to be indexed. For weighted MinHash, this
            is the sample size (`sample_size`).
        num_part (int): The number of partitions in LSH Ensemble.
        m (int): The memory usage factor: an LSH Ensemble uses approximately
            `m` times more memory space than a MinHash LSH with the same number of
            sets indexed. The higher the `m` the better the accuracy.
        weights (Tuple[float, float]): Used to adjust the relative importance of
            minizing false positive and false negative when optimizing
            for the Containment threshold. Similar to the `weights` parameter
            in :class:`datasketch.MinHashLSH`.
        storage_config (Optional[Dict]): Type of storage service to use for storing
            hashtables and keys.
            `basename` is an optional property whose value will be used as the prefix to
            stored keys. If this is not set, a random string will be generated instead. If you
            set this, you will be responsible for ensuring there are no key collisions.
        prepickle (Optional[bool]): If True, all keys are pickled to bytes before
            insertion. If None, a default value is chosen based on the
            `storage_config`.
        mergeable (bool): If True, the index keeps an immutable copy of every
            indexed set's hash values and size, so it can later be combined
            with another mergeable index using :meth:`merge`. This costs
            ``8 * num_perm`` bytes per indexed key. Indexes built without this
            option behave and are queried exactly as before and cannot be
            merged. (`default=False`)

    Note:
        Using more partitions (`num_part`) leads to better accuracy, at the
        expense of slower query performance.
        This is different from `the paper`_ and the `Go implementation`_, in which
        more partitions leads to better accuracy AND faster query performance,
        due to parallelism.

    Note:
        More information about the parameter `m` can be found in the
        `Go implementation`_
        of LSH Ensemble, in which `m` is named `MaxK`.

    .. _`Go implementation`: https://github.com/ekzhu/lshensemble
    .. _`the paper`: http://www.vldb.org/pvldb/vol9/p1185-zhu.pdf

    """

    def __init__(
        self,
        threshold: float = 0.9,
        num_perm: int = 128,
        num_part: int = 16,
        m: int = 8,
        weights: tuple[float, float] = (0.5, 0.5),
        storage_config: Optional[dict] = None,
        prepickle: Optional[bool] = None,
        mergeable: bool = False,
    ) -> None:
        if threshold > 1.0 or threshold < 0.0:
            raise ValueError("threshold must be in [0.0, 1.0]")
        if num_perm < 2:
            raise ValueError("Too few permutation functions")
        if num_part < 1:
            raise ValueError("num_part must be at least 1")
        if m < 2 or m > num_perm:
            raise ValueError("m must be in the range of [2, num_perm]")
        if any(w < 0.0 or w > 1.0 for w in weights):
            raise ValueError("Weight must be in [0.0, 1.0]")
        if sum(weights) != 1.0:
            raise ValueError("Weights must sum to 1.0")
        if not isinstance(mergeable, bool):
            raise TypeError("mergeable must be a bool")
        self.threshold = threshold
        self.h = num_perm
        self.m = m
        self._num_part = num_part
        self._weights = tuple(weights)
        self._mergeable = mergeable
        # Retained per-key state, only populated for mergeable indexes.
        # Each value is a ``_RetainedEntry`` whose hash values are immutable
        # bytes; the caller's MinHash arrays are never kept.
        self._entries = {} if mergeable else None
        self._seed = None
        self._permutations = None
        rs = self._init_optimal_params(weights)
        # Initialize multiple LSH indexes for each partition
        storage_config = storage_config if storage_config else {"type": "dict"}
        self._storage_type = storage_config["type"]
        basename = storage_config.get("basename", _random_name(11))
        self.indexes = [
            dict(
                (
                    r,
                    MinHashLSH(
                        num_perm=self.h,
                        params=(int(self.h / r), r),
                        storage_config=self._get_storage_config(basename, storage_config, partition, r),
                        prepickle=prepickle,
                    ),
                )
                for r in rs
            )
            for partition in range(0, num_part)
        ]
        self._prepickle = next(iter(self.indexes[0].values())).prepickle
        self.lowers = [None for _ in self.indexes]
        self.uppers = [None for _ in self.indexes]

    def _init_optimal_params(self, weights):
        false_positive_weight, false_negative_weight = weights
        self.xqs = np.exp(np.linspace(-5, 5, 10))
        self.params = np.array(
            [
                _optimal_param(
                    self.threshold,
                    self.h,
                    self.m,
                    xq,
                    false_positive_weight,
                    false_negative_weight,
                )
                for xq in self.xqs
            ],
            dtype=int,
        )
        # Find all unique r
        rs = set()
        for _, r in self.params:
            rs.add(r)
        return rs

    def _get_optimal_param(self, x, q):
        i = np.searchsorted(self.xqs, float(x) / float(q), side="left")
        if i == len(self.params):
            i = i - 1
        return self.params[i]

    def _get_storage_config(self, basename, base_config, partition, r):
        config = dict(base_config)
        config["basename"] = b"-".join([basename, struct.pack(">H", partition), struct.pack(">H", r)])
        return config

    def _entry_id(self, key: Hashable):
        """Return the form a key takes inside this index's storage.

        The default dict storage keeps keys as given unless ``prepickle`` is
        on, in which case they are pickled. Using that same form for the
        retained map keeps key equality identical to the one the LSH applies.
        """
        if self._prepickle:
            return pickle.dumps(key)
        return key

    @staticmethod
    def _sketch_coordinates(minhash):
        """Extract ``(hashvalues_bytes, seed, permutations)`` from a sketch.

        The hash values are copied into immutable ``bytes``; the returned
        value never aliases the caller's array.

        Raises:
            TypeError: If the sketch is not a plain :class:`MinHash` with
                permutation parameters (for example ``WeightedMinHash``,
                ``bBitMinHash`` or a ``LeanMinHash``).
            ValueError: If the sketch length is inconsistent.

        """
        if not isinstance(minhash, MinHash):
            raise TypeError(
                "MinHashLSHEnsemble only supports plain MinHash sketches, got %s" % type(minhash).__name__
            )
        permutations = getattr(minhash, "permutations", None)
        if permutations is None:
            raise TypeError(
                "MinHashLSHEnsemble requires a MinHash with permutation parameters, "
                "got %s without them" % type(minhash).__name__
            )
        # Cast to the same dtype the LSH band hashing assumes, so the retained
        # bytes are a faithful, immutable copy of the sketch's coordinates.
        hashvalues = np.asarray(minhash.hashvalues, dtype=np.uint64)
        snapshot = (
            np.asarray(permutations[0], dtype=np.uint64).tobytes(),
            np.asarray(permutations[1], dtype=np.uint64).tobytes(),
        )
        return hashvalues.tobytes(), minhash.seed, snapshot

    def _assign_partitions(self, entries):
        """Compute partition bounds from ``[(key, hashvalues_bytes, size)]``.

        The distribution is computed from the de-duplicated data that will
        actually be indexed, so a merged index has exactly the partitions a
        single-shot build of the same unique entries would have.
        """
        if len(entries) == 0:
            self.lowers = [None for _ in self.indexes]
            self.uppers = [None for _ in self.indexes]
            return
        sizes, counts = np.array(sorted(Counter(e[2] for e in entries).most_common())).T
        partitions = optimal_partitions(sizes, counts, len(self.indexes))
        for i, (lower, upper) in enumerate(partitions):
            self.lowers[i], self.uppers[i] = lower, upper

    def _index_prepared(self, entries):
        """Insert already-validated entries into the current partition bounds.

        Each key is inserted into exactly one partition -- the one whose
        bounds contain its size -- and, inside it, into every ``r``-level
        index. Nothing from any previous partitioning is touched, so no
        leftover bucket from an older layout can survive.

        Args:
            entries: Iterable of ``(key, hashvalues_bytes, size)``.

        """
        ordered = sorted(entries, key=lambda e: e[2])
        curr_part = 0
        last_part = len(self.indexes) - 1
        for key, hashvalues, size in ordered:
            while curr_part < last_part and size > self.uppers[curr_part]:
                curr_part += 1
            sketch = _RetainedSketch(hashvalues, self._seed)
            for r in self.indexes[curr_part]:
                self.indexes[curr_part][r].insert(key, sketch)

    def _build(self, prepared, seed, permutations):
        """Build this index from validated, de-duplicated entries.

        This is the same flow :meth:`index` uses: partition from the size
        histogram, then insert into exactly one partition per key. It keeps
        an immutable snapshot of each key's hash values and size so the index
        can be merged again later.

        Args:
            prepared (list): ``[(key, hashvalues_bytes, size)]``.
            seed (int): The permutation seed the bytes were generated with.
            permutations (tuple): Snapshot of the permutation parameters.

        """
        self._seed = seed
        self._permutations = permutations
        self._assign_partitions(prepared)
        self._index_prepared(prepared)
        self._entries = {
            self._entry_id(key): _RetainedEntry(key, hashvalues, size) for key, hashvalues, size in prepared
        }

    def index(self, entries: Iterable[tuple[Hashable, MinHash, int]]) -> None:
        """Index all sets given their keys, MinHashes, and sizes.
        It can be called only once after the index is created.

        Args:
            entries (Iterable[tuple[Hashable, MinHash, int]]): An iterable of
                tuples, each must be in the form of ``(key, minhash, size)``,
                where ``key`` is the unique
                identifier of a set, ``minhash`` is the MinHash of the set,
                and ``size`` is the size or number of unique items in the set.

        Raises:
            ValueError: If the index is not empty, ``entries`` is empty, a
                size is not positive, a key repeats, or (for a ``mergeable``
                index) a sketch's seed or permutation parameters differ from
                the first entry's.
            TypeError: If a ``mergeable`` index is given a sketch that is not
                a plain :class:`MinHash` with permutation parameters.

        """
        if not self.is_empty():
            raise ValueError("Cannot call index again on a non-empty index")

        if not self._mergeable:
            # Legacy path: unchanged behaviour and memory use.
            if not isinstance(entries, list):
                queue = deque()
                for key, minhash, size in entries:
                    if size <= 0:
                        raise ValueError("Set size must be positive")
                    queue.append((key, minhash, size))
                entries = list(queue)
            if len(entries) == 0:
                raise ValueError("entries is empty")
            sizes, counts = np.array(sorted(Counter(e[2] for e in entries).most_common())).T
            partitions = optimal_partitions(sizes, counts, len(self.indexes))
            for i, (lower, upper) in enumerate(partitions):
                self.lowers[i], self.uppers[i] = lower, upper
            entries.sort(key=lambda e: e[2])
            curr_part = 0
            for key, minhash, size in entries:
                u = self.uppers[curr_part]
                if size > u:
                    curr_part += 1
                for r in self.indexes[curr_part]:
                    self.indexes[curr_part][r].insert(key, minhash)
            return

        # Mergeable path: validate every sketch and snapshot the immutable
        # hash-value bytes before assigning partitions, so a bad sketch or a
        # coordinate mismatch leaves the index empty and untouched. Only the
        # per-key ``num_perm * 8`` byte snapshot is kept -- never a second
        # array of the caller's mutable hash values. If the build fails
        # halfway (allocation error, bad entry) nothing has been inserted into
        # this index yet.
        seen = set()
        prepared = []
        seed = None
        permutations = None
        for key, minhash, size in entries:
            if size <= 0:
                raise ValueError("Set size must be positive")
            entry_id = self._entry_id(key)
            if entry_id in seen:
                raise ValueError("The given key already exists: %r" % (key,))
            seen.add(entry_id)
            hashvalues, sketch_seed, sketch_permutations = self._sketch_coordinates(minhash)
            if len(hashvalues) // 8 != self.h:
                raise ValueError("Expecting minhash with length %d, got %d" % (self.h, len(hashvalues) // 8))
            if seed is None:
                seed, permutations = sketch_seed, sketch_permutations
            elif sketch_seed != seed:
                raise ValueError(
                    "Expecting minhash with seed %d, got %d: different permutations generate different coordinates"
                    % (seed, sketch_seed)
                )
            elif sketch_permutations != permutations:
                raise ValueError(
                    "Expecting minhash with the same permutation parameters as the index, got different parameters"
                )
            prepared.append((key, hashvalues, size))
        if not prepared:
            raise ValueError("entries is empty")
        self._build(prepared, seed, permutations)

    def merge(self, other: "MinHashLSHEnsemble", *, overlap: str = "identical") -> "MinHashLSHEnsemble":
        """Combine this index with `other` into a new, independent index.

        The two indexes were built from disjoint (or partly overlapping)
        shards of a collection. Their partitions were computed from each
        shard's own size distribution, so their bucket layouts are not
        compatible. The merge therefore rebuilds the partition bounds from
        the *combined, de-duplicated* size histogram and re-inserts every
        key into the partition its size falls into, using the same build flow
        as :meth:`index`. Queries against the result select the same bands
        and return the same candidate keys as a single-shot index built from
        the same unique entries (key order is unspecified).

        Neither source is modified: the result uses fresh bucket storage and
        a fresh key map, and existing paged cursors over either source can
        keep reading. This index and `other` must both have been created with
        ``mergeable=True``; otherwise :class:`MergeNotSupportedError` is
        raised before anything is read or written. The result is itself
        mergeable.

        Only the default in-process ``dict`` storage is supported. Other
        backends raise :class:`datasketch.storage.UnsupportedStorageError`
        before any index content is touched.

        Both indexes must share the same construction configuration:
        ``threshold``, ``num_perm``, ``num_part``, ``m``, the optimization
        ``weights``, and the key encoding (``prepickle``). In addition, the
        sketches must have been produced with the same ``seed`` and the same
        permutation parameters, so their hash values live in the same
        coordinate system. A shard that mixed coordinate systems is already
        rejected when it is indexed. These checks are made even for empty
        shards.

        Args:
            other (MinHashLSHEnsemble): The other mergeable index.
            overlap (str): How to treat keys present in both indexes.
                ``"identical"`` (default) accepts an overlapping key once,
                only if its size, hash coordinate system, and full
                ``hashvalues`` are identical; any difference raises
                :class:`ValueError`. ``"error"`` raises :class:`ValueError`
                as soon as any key overlaps. A key is never kept as two
                records, and an accepted overlap contributes to the size
                histogram only once.

        Returns:
            MinHashLSHEnsemble: A new, independently stored mergeable index.

        Raises:
            MergeNotSupportedError: If either index was not built with
                ``mergeable=True``.
            UnsupportedStorageError: If either index is not backed by the
                default dict storage.
            ValueError: If the construction configurations or sketch
                coordinate systems differ, `overlap` is not one of the
                supported strings, or an overlapping key conflicts under
                ``"identical"``.

        """
        if not isinstance(other, MinHashLSHEnsemble):
            raise TypeError("Can only merge with another MinHashLSHEnsemble, got %s" % type(other).__name__)
        # ``getattr`` with a False default keeps indexes unpickled from a build
        # that predates mergeable storage working as non-mergeable.
        if not getattr(self, "_mergeable", False) or not getattr(other, "_mergeable", False):
            raise MergeNotSupportedError(
                "Merging requires indexes built with mergeable=True; "
                "an index built without it cannot retain the per-key data a re-partition needs"
            )
        if overlap not in ("error", "identical"):
            raise ValueError("overlap must be 'error' or 'identical', got %r" % (overlap,))
        # Check the storage backend before touching any content. Both indexes
        # are only read below, but the result must be dict-backed too.
        for index in (self, other):
            storage_type = getattr(index, "_storage_type", None)
            if storage_type != "dict":
                raise UnsupportedStorageError("Merging requires the default dict storage, got %r" % (storage_type,))
        self._check_merge_compatible(other)

        merged = dict(self._entries)
        for entry_id, entry in other._entries.items():
            existing = merged.get(entry_id)
            if existing is None:
                merged[entry_id] = entry
                continue
            if overlap == "error":
                raise ValueError("The keys are overlapping, duplicate key exists: %r" % (entry.key,))
            if existing.size != entry.size or existing.hashvalues != entry.hashvalues:
                raise ValueError(
                    "Overlapping key %r has different sizes or hash values in the two indexes" % (entry.key,)
                )
            # Accepted identical overlap: keep one copy, so the histogram
            # counts this key once.

        result = MinHashLSHEnsemble(
            threshold=self.threshold,
            num_perm=self.h,
            num_part=self._num_part,
            m=self.m,
            weights=self._weights,
            storage_config={"type": "dict"},
            prepickle=self._prepickle,
            mergeable=True,
        )
        prepared = [(entry.key, entry.hashvalues, entry.size) for entry in merged.values()]
        if prepared:
            seed = self._seed if self._seed is not None else other._seed
            permutations = self._permutations if self._permutations is not None else other._permutations
            result._build(prepared, seed, permutations)
        return result

    def _check_merge_compatible(self, other: "MinHashLSHEnsemble") -> None:
        """Validate that two mergeable indexes share build configuration.

        This never reads bucket storage; it only compares configuration held
        on the objects, so it works for empty shards too.
        """
        problems = []
        if self.threshold != other.threshold:
            problems.append("threshold (%r != %r)" % (self.threshold, other.threshold))
        if self.h != other.h:
            problems.append("num_perm (%r != %r)" % (self.h, other.h))
        if self._num_part != other._num_part:
            problems.append("num_part (%r != %r)" % (self._num_part, other._num_part))
        if self.m != other.m:
            problems.append("m (%r != %r)" % (self.m, other.m))
        if self._weights != other._weights:
            problems.append("weights (%r != %r)" % (self._weights, other._weights))
        if self._prepickle != other._prepickle:
            problems.append("prepickle/key encoding (%r != %r)" % (self._prepickle, other._prepickle))
        if problems:
            raise ValueError(
                "Cannot merge MinHashLSHEnsemble indexes with different configuration: " + ", ".join(problems)
            )
        if self._seed is not None and other._seed is not None:
            if self._seed != other._seed:
                raise ValueError(
                    "Cannot merge indexes whose sketches used different seeds (%r != %r)"
                    % (self._seed, other._seed)
                )
            if self._permutations != other._permutations:
                raise ValueError("Cannot merge indexes whose sketches used different permutation parameters")

    def query(self, minhash: MinHash, size: int) -> Generator[Hashable, None, None]:
        """Giving the MinHash and size of the query set, retrieve
        keys that references sets with containment with respect to
        the query set greater than the threshold.

        Args:
            minhash (MinHash): The MinHash of the query set.
            size (int): The size (number of unique items) of the query set.

        Returns:
            Generator[Hashable, None, None]: an iterator of keys.

        """
        for i, index in enumerate(self.indexes):
            u = self.uppers[i]
            if u is None:
                continue
            b, r = self._get_optimal_param(u, size)
            for key in index[r]._query_b(minhash, b):
                yield key

    def query_paged(self, minhash: MinHash, size: int) -> "MinHashLSHEnsembleQuery":
        """Start a resumable, budgeted query over the candidates for a set.

        Unlike :meth:`query`, which yields every candidate in one pass, this
        returns a cursor whose :meth:`MinHashLSHEnsembleQuery.next_page` reads
        a bounded number of candidates per call. A caller can inspect the
        candidates of one page (for example, re-check those materials against
        the query) and then return for the next page without restarting the
        query or re-reading buckets that were already visited.

        The index's partitioning and per-partition band parameters are selected
        exactly as in :meth:`query`, using the query set size. Pages deliver
        each key at most once per query; read to the end, the concatenated
        pages hold the same key set as this fixed index's :meth:`query`.

        A page spends one budget unit for every bucket lookup (an empty bucket
        still counts) and one for every key actually read out of a bucket
        (including keys already delivered by another band). It stops as soon
        as the budget runs out and never copies a whole bucket up front.

        Use :meth:`MinHashLSHEnsembleQuery.close` (or a ``with`` block) to end
        the query early; afterwards no page can be read. Paged queries require
        bucket storage that supports lazy, resumable reads, which is checked
        before any bucket is accessed -- the default ``dict`` storage does.

        Args:
            minhash (MinHash): The MinHash of the query set.
            size (int): The size (number of unique items) of the query set.

        Returns:
            MinHashLSHEnsembleQuery: The paged query cursor.

        Raises:
            ValueError: If the MinHash length does not match the index, if
                `size` is not a positive integer, or if the bucket storage does
                not support paged scans.

        """
        return MinHashLSHEnsembleQuery(self, minhash, size)

    def __contains__(self, key: Hashable) -> bool:
        """Args:
            key (hashable): The unique identifier of a set.

        Returns:
            bool: True only if the key exists in the index.

        """
        return any(any(key in index[r] for r in index) for index in self.indexes)

    def is_empty(self) -> bool:
        """Returns:
        bool: Check if the index is empty.

        """
        return all(all(index[r].is_empty() for r in index) for index in self.indexes)


if __name__ == "__main__":
    import numpy as np

    xqs = np.exp(np.linspace(-5, 5, 10))
    threshold = 0.5
    max_r = 8
    num_perm = 256
    false_negative_weight, false_positive_weight = 0.5, 0.5
    for xq in xqs:
        b, r = _optimal_param(threshold, num_perm, max_r, xq, false_positive_weight, false_negative_weight)
        logger.info("threshold: %.2f, xq: %.3f, b: %d, r: %d", threshold, xq, b, r)
