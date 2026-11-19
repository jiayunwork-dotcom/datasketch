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
from datasketch.storage import UnsupportedStorageError, _random_name, supports_paged_bucket_scan

logger = logging.getLogger(__name__)


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
        self.threshold = threshold
        self.h = num_perm
        self.m = m
        rs = self._init_optimal_params(weights)
        # Initialize multiple LSH indexes for each partition
        storage_config = storage_config if storage_config else {"type": "dict"}
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
            ValueError: If the index is not empty or ``entries`` is empty.

        """
        if not self.is_empty():
            raise ValueError("Cannot call index again on a non-empty index")
        if not isinstance(entries, list):
            queue = deque()
            for key, minhash, size in entries:
                if size <= 0:
                    raise ValueError("Set size must be positive")
                queue.append((key, minhash, size))
            entries = list(queue)
        if len(entries) == 0:
            raise ValueError("entries is empty")
        # Create optimal partitions.
        sizes, counts = np.array(sorted(Counter(e[2] for e in entries).most_common())).T
        partitions = optimal_partitions(sizes, counts, len(self.indexes))
        for i, (lower, upper) in enumerate(partitions):
            self.lowers[i], self.uppers[i] = lower, upper
        # Insert into partitions.
        entries.sort(key=lambda e: e[2])
        curr_part = 0
        for key, minhash, size in entries:
            u = self.uppers[curr_part]
            if size > u:
                curr_part += 1
            for r in self.indexes[curr_part]:
                self.indexes[curr_part][r].insert(key, minhash)

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
