from __future__ import annotations

import contextlib
import pickle
import struct
from collections.abc import Hashable
from typing import Callable, List, Optional, Union

from scipy.integrate import quad as integrate

from datasketch.minhash import MinHash
from datasketch.storage import (
    DictListStorage,
    OrderedStorage,
    UnorderedStorage,
    UnsupportedStorageError,
    _random_name,
    ordered_storage,
    supports_paged_bucket_scan,
    unordered_storage,
)
from datasketch.weighted_minhash import WeightedMinHash


# Sentinel used by ``MinHashLSH.merge`` to tell "the caller did not use the
# new overlap option" apart from "the caller explicitly asked for a policy".
class _OverlapUnset:
    def __repr__(self) -> str:
        return "<unset>"


_OVERLAP_UNSET = _OverlapUnset()


def _false_positive_probability(threshold, b, r):
    _probability = lambda s: 1 - (1 - s ** float(r)) ** float(b)
    a, _err = integrate(_probability, 0.0, threshold)
    return a


def _false_negative_probability(threshold, b, r):
    _probability = lambda s: 1 - (1 - (1 - s ** float(r)) ** float(b))
    a, _err = integrate(_probability, threshold, 1.0)
    return a


def _optimal_param(threshold, num_perm, false_positive_weight, false_negative_weight):
    """Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
    of probabilities of false positive and false negative.
    """
    min_error = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        max_r = int(num_perm / b)
        for r in range(1, max_r + 1):
            fp = _false_positive_probability(threshold, b, r)
            fn = _false_negative_probability(threshold, b, r)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt


class MinHashLSH:
    """The :ref:`minhash_lsh` index.
    It supports query with `Jaccard similarity`_ threshold.
    Reference: `Chapter 3, Mining of Massive Datasets
    <http://www.mmds.org/>`_.

    Args:
        threshold (float): The Jaccard similarity threshold between 0.0 and
            1.0. The initialized MinHash LSH will be optimized for the threshold by
            minizing the false positive and false negative.
        num_perm (int): The number of permutation functions used
            by the MinHash to be indexed. For weighted MinHash, this
            is the sample size (`sample_size`).
        weights (Tuple[float, float]): Used to adjust the relative importance of
            minimizing false positive and false negative when optimizing
            for the Jaccard similarity threshold.
            `weights` is a tuple in the format of
            :code:`(false_positive_weight, false_negative_weight)`.
        params (Optiona[Tuple[int, int]]): The LSH parameters (i.e., number of bands and size
            of each bands). This is used to bypass the parameter optimization
            step in the constructor. `threshold` and `weights` will be ignored
            if this is given.
        storage_config (Optional[Dict]): Type of storage service to use for storing
            hashtables and keys.
            `basename` is an optional property whose value will be used as the prefix to
            stored keys. If this is not set, a random string will be generated instead. If you
            set this, you will be responsible for ensuring there are no key collisions.
        prepickle (Optional[bool]): If True, all keys are pickled to bytes before
            insertion. If not specified, a default value is chosen based on the
            `storage_config`.
        hashfunc (Optional[Callable[[bytes], bytes]]): If a hash function is provided it will be used to
            compress the index keys to reduce the memory footprint. This could cause a higher
            false positive rate.

    Note:
        `weights` must sum to 1.0, and the format is
        (false positive weight, false negative weight).
        For example, if minimizing false negative (or maintaining high recall) is more
        important, assign more weight toward false negative: weights=(0.4, 0.6).
        Try to live with a small difference between weights (i.e. < 0.5).

    Examples:
        Create an index with 128 permutation functions optimized for Jaccard
        threshold 0.9:

        .. code-block:: python

            lsh = MinHashLSH(threshold=0.9, num_perm=128)
            print(lsh.b, lsh.r)
            # 5 25

        The built-in optimizer will try to minimize the weighted sum of
        probabilities of false positive and false negative. The algorithm is
        a simple grid search over the space of possible parameters.

        Note that it is possible to get :attr:`b` (number of bands) and
        :attr:`r` (band size) that do not sum to :attr:`num_perm`, leading to
        unused permutation values in the indexed MinHash.
        This is because the optimizer only considers bands of
        the same size, and the number of bands is not necessarily a divisor of
        :attr:`num_perm`.

        Instead of using the built-in optimizer, you can customize the LSH
        parameters your self. The snippet below creates an index with 128
        permutation functions and 16 bands each with size 8, skipping the
        optimization step:

        .. code-block:: python

            lsh = MinHashLSH(num_perm=128, params=(16, 8))
            print(lsh.b, lsh.r)
            # 16 8

        Create an index backed by Redis storage:

        .. code-block:: python

            lsh = MinHashLSH(
                threshold=0.9,
                num_perm=128,
                storage_config={
                    "type": "redis",
                    "basename": b"mylsh",  # optional, defaults to a random string.
                    "redis": {"host": "localhost", "port": 6379},
                },
            )

        The `basename` property is optional. It is used to generate key prefixes
        in the storage layer to uniquely identify data associated with this LSH.
        Thus, if you create a new LSH object with the same `basename`, you will
        be using the same underlying data in the storage layer associated with
        a previous LSH object. If you do not set this property, a random string
        will be generated instead.

    """

    def __init__(
        self,
        threshold: float = 0.9,
        num_perm: int = 128,
        weights: tuple[float, float] = (0.5, 0.5),
        params: Optional[tuple[int, int]] = None,
        storage_config: Optional[dict] = None,
        prepickle: Optional[bool] = None,
        hashfunc: Optional[Callable[[bytes], bytes]] = None,
    ) -> None:
        storage_config = storage_config if storage_config else {"type": "dict"}
        self._buffer_size = 50000
        if threshold > 1.0 or threshold < 0.0:
            raise ValueError("threshold must be in [0.0, 1.0]")
        if num_perm < 2:
            raise ValueError("Too few permutation functions")
        if any(w < 0.0 or w > 1.0 for w in weights):
            raise ValueError("Weight must be in [0.0, 1.0]")
        if sum(weights) != 1.0:
            raise ValueError("Weights must sum to 1.0")
        self.h = num_perm
        if params is not None:
            self.b, self.r = params
            if self.b * self.r > num_perm:
                raise ValueError(
                    "The product of b and r in params is "
                    f"{self.b} * {self.r} = {self.b * self.r} -- it must be less than num_perm {num_perm}. "
                    "Did you forget to specify num_perm?"
                )
        else:
            false_positive_weight, false_negative_weight = weights
            self.b, self.r = _optimal_param(threshold, num_perm, false_positive_weight, false_negative_weight)
        if self.b < 2:
            raise ValueError("The number of bands are too small (b < 2)")

        self.prepickle = storage_config["type"] == "redis" if prepickle is None else prepickle
        self._require_bytes_keys = not (storage_config["type"] == "dict" or self.prepickle)

        self.hashfunc = hashfunc
        if hashfunc:
            self._H = self._hashed_byteswap
        else:
            self._H = self._byteswap

        basename = storage_config.get("basename", _random_name(11))
        self.hashtables: List[UnorderedStorage] = [
            unordered_storage(
                storage_config,
                name=b"".join([basename, b"_bucket_", struct.pack(">H", i)]),
            )
            for i in range(self.b)
        ]
        self.hashranges = [(i * self.r, (i + 1) * self.r) for i in range(self.b)]
        self.keys: OrderedStorage = ordered_storage(storage_config, name=b"".join([basename, b"_keys"]))

    @property
    def buffer_size(self) -> int:
        return self._buffer_size

    @buffer_size.setter
    def buffer_size(self, value: int) -> None:
        self.keys.buffer_size = value
        for t in self.hashtables:
            t.buffer_size = value
        self._buffer_size = value

    def insert(
        self,
        key: Hashable,
        minhash: Union[MinHash, WeightedMinHash],
        check_duplication: bool = True,
    ):
        """Insert a key to the index, together with a MinHash or Weighted MinHash
        of the set referenced by the key.

        Args:
            key (Hashable): The unique identifier of the set.
            minhash (Union[MinHash, WeightedMinHash]): The MinHash of the set.
            check_duplication (bool): To avoid duplicate keys in the storage
                (`default=True`). It's recommended to not change the default, but
                if you want to avoid the overhead during insert you can set
                `check_duplication = False`.

        """
        self._insert(key, minhash, check_duplication=check_duplication, buffer=False)

    def merge(
        self,
        other: MinHashLSH,
        check_overlap: bool = False,
        *,
        overlap: Optional[str] = _OVERLAP_UNSET,
    ):
        """Merge the other MinHashLSH with this one, making this one the union
        of both.

        Note:
            Only num_perm, number of bands and sizes of each band is checked for equivalency of two MinHashLSH indexes.
            Other initialization parameters threshold, weights, storage_config, prepickle and hash_func are not checked.

        Args:
            other (MinHashLSH): The other MinHashLSH.
            check_overlap (bool): Check if there are any overlapping keys before merging and raise if there are any.
                (`default=False`)
            overlap (Optional[str]): The policy for keys present in both
                indexes. When not passed, the legacy behaviour is kept. Pass
                ``"error"`` to raise as soon as any key overlaps, or
                ``"identical"`` to accept an overlapping key only once,
                provided every band signature stored for that key in this
                index equals the one stored in `other`; otherwise a
                :class:`ValueError` is raised. The comparison uses the
                signatures actually stored in each index, for all of that
                index's bands, so a key whose stored signature was changed in
                either index is rejected. This option cannot be combined with
                `check_overlap=True`.

        Raises:
            ValueError: If the two MinHashLSH have different initialization
                parameters, if `check_overlap` is `True` and there are
                overlapping keys, if an explicit `overlap` policy is combined
                with `check_overlap=True`, or if the `"identical"` policy finds
                an overlapping key with differing stored signatures.
            UnsupportedStorageError: If an `overlap` policy is requested for a
                bucket backend other than the default dict storage. The check
                happens before this index is modified.

        """
        if overlap is _OVERLAP_UNSET:
            self._merge(other, check_overlap=check_overlap, buffer=False)
            return
        if overlap not in ("error", "identical"):
            raise ValueError("overlap must be 'error' or 'identical', got %r" % (overlap,))
        if check_overlap:
            raise ValueError("check_overlap=True cannot be combined with the overlap option")
        self._merge(other, check_overlap=False, buffer=False, overlap=overlap)

    def insertion_session(self, buffer_size: int = 50000) -> MinHashLSHInsertionSession:
        """Create a context manager for fast insertion into this index.

        Args:
        e         buffer_size (int): The buffer size for insert_session mode (default=50000).

        Returns:
                 MinHashLSHInsertionSession: The context manager.

        Example:
                 Insert 100 MinHashes into an Redis-backed index using a session:

                 .. code-block:: python

                     from datasketch import MinHash, MinHashLSH
                     import numpy as np

                     minhashes = []
                     for i in range(100):
                         m = MinHash(num_perm=128)
                         m.update_batch(np.random.randint(low=0, high=30, size=10))
                         minhashes.append(m)

                     lsh = MinHashLSH(
                         threshold=0.5,
                         num_perm=128,
                         storage_config={
                             "type": "redis",
                             "redis": {"host": "localhost", "port": 6379},
                         },
                     )
                     with lsh.insertion_session() as session:
                         for i, m in enumerate(minhashes):
                             session.insert(i, m)

        """
        return MinHashLSHInsertionSession(self, buffer_size=buffer_size)

    def deletion_session(self, buffer_size: int = 50000) -> MinHashLSHDeletionSession:
        """Create a context manager for fast deletion from this index.

        Args:
                buffer_size (int): The buffer size for deletion operations (default=50000).

        Returns:
                MinHashLSHDeletionSession: The context manager.

        r    Example:
                Delete keys from a Redis-backed index using a session:

                .. code-block:: python

                    from datasketch import MinHashLSH

                    lsh = MinHashLSH(
                        threshold=0.5,
                        num_perm=128,
                        storage_config={
                            "type": "redis",
                            "redis": {"host": "localhost", "port": 6379},
                        },
                    )

                    # ... insert some data ...

                    keys_to_delete = ["key1", "key2", "key3"]
                    with lsh.deletion_session() as session:
                        for key in keys_to_delete:
                            session.remove(key)

        """
        return MinHashLSHDeletionSession(self, buffer_size=buffer_size)

    def _insert(
        self,
        key: Hashable,
        minhash: Union[MinHash, WeightedMinHash],
        check_duplication: bool = True,
        buffer: bool = False,
    ):
        if len(minhash) != self.h:
            raise ValueError("Expecting minhash with length %d, got %d" % (self.h, len(minhash)))
        if self._require_bytes_keys and not isinstance(key, bytes):
            raise TypeError(
                f"prepickle=False requires bytes keys for non-dict storage, got {type(key).__name__}. "
                "Either pass bytes keys or use prepickle=True for automatic serialization."
            )
        if self.prepickle:
            key = pickle.dumps(key)
        if check_duplication and key in self.keys:
            raise ValueError("The given key already exists")
        Hs = [self._H(minhash.hashvalues[start:end]) for start, end in self.hashranges]
        self.keys.insert(key, *Hs, buffer=buffer)
        for H, hashtable in zip(Hs, self.hashtables):
            hashtable.insert(H, key, buffer=buffer)

    def __equivalent(self, other: MinHashLSH) -> bool:
        """Returns:
        bool: If the two MinHashLSH have equal num_perm, number of bands, size of each band then two are equivalent.

        """
        return type(self) is type(other) and self.h == other.h and self.b == other.b and self.r == other.r

    def _merge(
        self,
        other: MinHashLSH,
        check_overlap: bool = False,
        buffer: bool = False,
        overlap=_OVERLAP_UNSET,
    ) -> None:
        if overlap is _OVERLAP_UNSET:
            # Legacy path: kept byte-for-byte compatible.
            if self.__equivalent(other):
                if check_overlap and set(self.keys).intersection(set(other.keys)):
                    raise ValueError("The keys are overlapping, duplicate key exists.")
                for key in other.keys:
                    Hs = other.keys.get(key)
                    self.keys.insert(key, *Hs, buffer=buffer)
                    for H, hashtable in zip(Hs, self.hashtables):
                        hashtable.insert(H, key, buffer=buffer)
            else:
                if type(self) is not type(other):
                    raise ValueError(f"Cannot merge type MinHashLSH and type {type(other).__name__}.")
                raise ValueError("Cannot merge MinHashLSH with different initialization parameters.")
            return

        if overlap not in ("error", "identical"):
            raise ValueError("overlap must be 'error' or 'identical', got %r" % (overlap,))
        if buffer:
            raise ValueError("An explicit overlap policy cannot be used with buffered merging")
        if not self.__equivalent(other):
            if type(self) is not type(other):
                raise ValueError(f"Cannot merge type MinHashLSH and type {type(other).__name__}.")
            raise ValueError("Cannot merge MinHashLSH with different initialization parameters.")
        # The in-place rollback below relies on the two dict-backed containers.
        for lsh in (self, other):
            storages = [lsh.keys, *lsh.hashtables]
            if not all(isinstance(storage, DictListStorage) for storage in storages):
                raise UnsupportedStorageError(
                    "Overlap-aware merging requires the default dict storage for both indexes, got %s"
                    % type(lsh.keys).__name__
                )

        # Plan every write before touching this index, so a conflict at the
        # end of one index cannot leave earlier non-conflicting keys behind.
        plan = []
        planned_ids = set()
        for key in list(other.keys):
            Hs = list(other.keys.get(key))
            if key in self.keys:
                if overlap == "error":
                    raise ValueError("The keys are overlapping, duplicate key exists.")
                # "identical": accept a key already present once only if every
                # stored band signature matches. Comparing the whole stored
                # signature list, not a prefix, catches a changed late band.
                existing = list(self.keys.get(key))
                if existing != Hs:
                    raise ValueError(
                        "Cannot merge keys with identical identifiers but different stored signatures: %r" % (key,)
                    )
                # Accepted identical overlap: the key is already present once,
                # so no bucket or key-map write is needed.
                continue
            if key in planned_ids:
                # The other index itself stores this key more than once. Both
                # copies are identical by construction (same key => same
                # buckets), so a second record would only duplicate it.
                continue
            planned_ids.add(key)
            plan.append((key, Hs))

        if not plan:
            return

        # Apply, recording enough to undo a partial write. Each write is
        # recorded *before* it happens, because a failing backend write may
        # itself leave a partial value behind.
        added_keys = []
        added_buckets = []
        try:
            for key, Hs in plan:
                for H, hashtable in zip(Hs, self.hashtables):
                    if key not in hashtable.get(H):
                        added_buckets.append((hashtable, H, key))
                        hashtable.insert(H, key)
                # Recorded before the write so a partial key-map write is also
                # removed on rollback; the key was proven absent above.
                added_keys.append(key)
                self.keys.insert(key, *Hs)
        except BaseException:
            for hashtable, H, key in reversed(added_buckets):
                with contextlib.suppress(KeyError, ValueError):
                    hashtable.remove_val(H, key)
                if not hashtable.get(H) and hashtable.has_key(H):
                    hashtable.remove(H)
            for key in reversed(added_keys):
                if key in self.keys:
                    self.keys.remove(key)
            raise

    def query(self, minhash) -> list[Hashable]:
        """Giving the MinHash of the query set, retrieve
        the keys that reference sets with Jaccard
        similarities likely greater than the threshold.

        Results are based on minhash segment collision
        and are thus approximate. For more accurate results,
        filter again with :meth:`MinHash.jaccard`. For exact results,
        filter by computing Jaccard similarity using original sets.

        Args:
            minhash (MinHash): The MinHash of the query set.

        Returns:
            list: a list of unique keys.

        Example:
            Query and rank results using :meth:`MinHash.jaccard`.

            .. code-block:: python

                from datasketch import MinHash, MinHashLSH
                import numpy as np

                # Generate 100 random MinHashes.
                minhashes = MinHash.bulk(np.random.randint(low=0, high=30, size=(100, 10)), num_perm=128)

                # Create LSH index.
                lsh = MinHashLSH(threshold=0.5, num_perm=128)
                for i, m in enumerate(minhashes):
                    lsh.insert(i, m)

                # Get the initial results from LSH.
                query = minhashes[0]
                results = lsh.query(query)

                # Rank results using Jaccard similarity estimated by MinHash.
                results = [(query.jaccard(minhashes[key]), key) for key in results]
                results.sort(reverse=True)
                print(results)

            Output:

            .. code-block::

                [(1.0, 0), (0.421875, 4), (0.4140625, 19), (0.359375, 58),
                 (0.3359375, 78), (0.265625, 62), (0.2578125, 11), (0.25, 98), (0.171875, 21)]

            Note that although the threshold is set to 0.5, the results are not
            guaranteed to be above 0.5 because the LSH index is approximate and
            the Jaccard similarity is estimated by MinHash.

        """
        if len(minhash) != self.h:
            raise ValueError("Expecting minhash with length %d, got %d" % (self.h, len(minhash)))
        candidates = set()
        for (start, end), hashtable in zip(self.hashranges, self.hashtables):
            H = self._H(minhash.hashvalues[start:end])
            for key in hashtable.get(H):
                candidates.add(key)
        if self.prepickle:
            return [pickle.loads(key) for key in candidates]
        return list(candidates)

    def add_to_query_buffer(self, minhash: Union[MinHash, WeightedMinHash]) -> None:
        """Giving the MinHash of the query set, buffer
        queries to retrieve the keys that references
        sets with Jaccard similarities greater than
        the threshold.

        Buffered queries can be executed using
        :meth:`collect_query_buffer`. The combination of these
        functions is way faster if cassandra backend
        is used with `shared_buffer`.

        Args:
            minhash (MinHash): The MinHash of the query set.

        """
        if len(minhash) != self.h:
            raise ValueError("Expecting minhash with length %d, got %d" % (self.h, len(minhash)))
        for (start, end), hashtable in zip(self.hashranges, self.hashtables):
            H = self._H(minhash.hashvalues[start:end])
            hashtable.add_to_select_buffer([H])

    def collect_query_buffer(self) -> list[Hashable]:
        """Execute and return buffered queries given
        by :meth:`add_to_query_buffer`.

        If multiple query MinHash were added to the query buffer,
        the intersection of the results of all query MinHash will be returned.

        Returns:
            list: a list of unique keys.

        """
        collected_result_sets = [
            set(collected_result_lists)
            for hashtable in self.hashtables
            for collected_result_lists in hashtable.collect_select_buffer()
        ]
        if not collected_result_sets:
            return []
        if self.prepickle:
            return [pickle.loads(key) for key in set.intersection(*collected_result_sets)]
        return list(set.intersection(*collected_result_sets))

    def __contains__(self, key: Hashable) -> bool:
        """Args:
            key (Hashable): The unique identifier of a set.

        Returns:
            bool: True only if the key exists in the index.

        """
        if self.prepickle:
            key = pickle.dumps(key)
        return key in self.keys

    def remove(self, key: Hashable) -> None:
        """Remove the key from the index.

        Args:
            key (Hashable): The unique identifier of a set.

        Raises:
            ValueError: If the key does not exist.

        """
        self._remove(key, buffer=False)

    def _remove(self, key: Hashable, buffer: bool = False) -> None:
        """Internal remove method with optional buffering support.

        Args:
            key (Hashable): The unique identifier of a set.
            buffer (bool): If True, buffer the removal operations.

        Raises:
            ValueError: If the key does not exist.

        """
        if self.prepickle:
            key = pickle.dumps(key)
        if key not in self.keys:
            raise ValueError("The given key does not exist")
        for H, hashtable in zip(self.keys[key], self.hashtables):
            hashtable.remove_val(H, key, buffer=buffer)
            if not hashtable.get(H):
                hashtable.remove(H, buffer=buffer)
        self.keys.remove(key, buffer=buffer)

    def is_empty(self) -> bool:
        """Returns:
        bool: `True` only if the index is empty.

        """
        return any(t.size() == 0 for t in self.hashtables)

    def _byteswap(self, hs):
        return bytes(hs.byteswap().data)

    def _hashed_byteswap(self, hs):
        if self.hashfunc is None:
            raise RuntimeError("Hash function not configured.")
        return self.hashfunc(bytes(hs.byteswap().data))

    def _query_b(self, minhash, b):
        if len(minhash) != self.h:
            raise ValueError("Expecting minhash with length %d, got %d" % (self.h, len(minhash)))
        if b > len(self.hashtables):
            raise ValueError("b must be less or equal to the number of hash tables")
        candidates = set()
        for (start, end), hashtable in zip(self.hashranges[:b], self.hashtables[:b]):
            H = self._H(minhash.hashvalues[start:end])
            if H in hashtable:
                for key in hashtable[H]:
                    candidates.add(key)
        if self.prepickle:
            return {pickle.loads(key) for key in candidates}
        return candidates

    def _paged_bucket_lookup(self, hashvalues, band: int):
        """Look up the bucket of one band without reading any of its keys.

        This is the metered "bucket lookup" step of a paged query: it computes
        the band's bucket key and asks the bucket storage how many keys it
        holds, but never copies the bucket. An absent bucket and an empty
        bucket are equivalent and both count as a lookup.

        Args:
            hashvalues (numpy.ndarray): The hash values of the query MinHash.
            band (int): Index of the band (hash table) to look up.

        Returns:
            tuple: ``(bucket_key, bucket_size)``. ``bucket_size`` is 0 when the
            bucket is absent or empty.

        Raises:
            ValueError: If `band` is out of range.
            datasketch.storage.UnsupportedStorageError: If the bucket storage
                cannot iterate a bucket lazily.

        """
        if band < 0 or band >= len(self.hashtables):
            raise ValueError("band must be in [0, %d)" % len(self.hashtables))
        hashtable = self.hashtables[band]
        if not supports_paged_bucket_scan(hashtable):
            raise UnsupportedStorageError(
                "Paged bucket scans require dict-backed bucket storage, got %s" % type(hashtable).__name__
            )
        start, end = self.hashranges[band]
        H = self._H(hashvalues[start:end])
        return H, hashtable.bucket_size(H)

    def _paged_bucket_cursor(self, band: int, bucket_key):
        """Return a lazy, resumable cursor over a band's bucket.

        Only one key is produced per ``next()`` call and the cursor keeps its
        position, so a caller can stop after a budget is spent and continue
        from the same position on a later page without the bucket being copied.

        Args:
            band (int): Index of the band (hash table).
            bucket_key: The bucket key returned by
                :meth:`MinHashLSH._paged_bucket_lookup`.

        Returns:
            Iterator: A cursor over the bucket's keys.

        Raises:
            datasketch.storage.UnsupportedStorageError: If the bucket storage
                cannot be scanned lazily.

        """
        hashtable = self.hashtables[band]
        if not supports_paged_bucket_scan(hashtable):
            raise UnsupportedStorageError(
                "Paged bucket scans require dict-backed bucket storage, got %s" % type(hashtable).__name__
            )
        return hashtable.bucket_cursor(bucket_key)

    def get_counts(self) -> list[dict[Hashable, int]]:
        """Returns a list of length :attr:`b` (i.e., number of hash tables) with
        each element a dictionary mapping hash table bucket key to the number
        of indexed keys stored under each bucket.

        Returns:
            list: a list of dictionaries.

        """
        return [hashtable.itemcounts() for hashtable in self.hashtables]

    def get_subset_counts(self, *keys: Hashable) -> list[dict[Hashable, int]]:
        """Returns the bucket allocation counts (see :meth:`get_counts` above)
        restricted to the list of keys given.

        Args:
            keys (Hashable) : the keys for which to get the bucket allocation
                counts.

        Returns:
            list: a list of dictionaries.

        """
        key_set = [pickle.dumps(key) for key in set(keys)] if self.prepickle else list(set(keys))
        hashtables = [unordered_storage({"type": "dict"}) for _ in range(self.b)]
        Hss = self.keys.getmany(*key_set)
        for key, Hs in zip(key_set, Hss):
            for H, hashtable in zip(Hs, hashtables):
                hashtable.insert(H, key)
        return [hashtable.itemcounts() for hashtable in hashtables]


class MinHashLSHInsertionSession:
    """Context manager for batch insertion of documents into a MinHashLSH.

    Args:
        lsh (MinHashLSH): The MinHashLSH to insert into.
        buffer_size (int): The buffer size for insert_session mode.

    """

    def __init__(self, lsh: MinHashLSH, buffer_size: int):
        self.lsh = lsh
        self.lsh.buffer_size = buffer_size

    def __enter__(self) -> MinHashLSHInsertionSession:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        self.lsh.keys.empty_buffer()
        for hashtable in self.lsh.hashtables:
            hashtable.empty_buffer()

    def insert(
        self,
        key: Hashable,
        minhash: Union[MinHash, WeightedMinHash],
        check_duplication=True,
    ) -> None:
        """Insert a unique key to the index, together
        with a MinHash (or weighted MinHash) of the set referenced by
        the key.

        Args:
            key (Hashable): The unique identifier of the set.
            minhash (Union[MinHash, WeightedMinhash]): The MinHash of the set.

        """
        self.lsh._insert(key, minhash, check_duplication=check_duplication, buffer=True)


class MinHashLSHDeletionSession:
    """Context manager for batch deletion of documents from a MinHashLSH.

    Args:
        lsh (MinHashLSH): The MinHashLSH to delete from.
        buffer_size (int): The buffer size for deletion operations.

    """

    def __init__(self, lsh: MinHashLSH, buffer_size: int):
        self.lsh = lsh
        self.lsh.buffer_size = buffer_size

    def __enter__(self) -> MinHashLSHDeletionSession:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        self.lsh.keys.empty_buffer()
        for hashtable in self.lsh.hashtables:
            hashtable.empty_buffer()

    def remove(self, key: Hashable) -> None:
        """Remove a key from the index.

        Args:
            key (Hashable): The unique identifier to remove.

        Raises:
            ValueError: If the key does not exist.

        """
        self.lsh._remove(key, buffer=True)
