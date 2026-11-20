"""Tests for combining MinHashLSHEnsemble shards into a re-partitioned index.

The fixtures here deliberately use data whose partition membership changes
when the shards are combined: one shard holds small sets, the other holds much
larger sets, so each shard's own size histogram yields different partitions
from the combined one. A merged index is checked entry by entry against an
index built in a single shot from the same unique entries.
"""

import unittest
from unittest import mock

import numpy as np

from datasketch import (
    LeanMinHash,
    MergeNotSupportedError,
    MinHash,
    MinHashLSHEnsemble,
)
from datasketch.storage import UnsupportedStorageError
from datasketch.weighted_minhash import WeightedMinHashGenerator


def _minhash(values, num_perm=16, seed=1):
    m = MinHash(num_perm=num_perm, seed=seed)
    m.update_batch(values)
    return m


def _sketch(key, size, num_perm=16, seed=1):
    """Deterministic MinHash of `size` distinct tokens related to `key`.

    A shared token pool is used so that queries against these sketches return
    real candidates; the key-specific tokens keep sketches distinct.
    """
    tokens = [b"pool-%d" % (i % 64) for i in range(min(size, 64))]
    tokens += [("%s-token-%d" % (key, i)).encode() for i in range(min(size, 8))]
    return _minhash(tokens, num_perm=num_perm, seed=seed)


# Sizes of the two shards. `_SMALL` lies far below `_LARGE`, so each shard's
# four partitions cover disjoint ranges while the combined data does not.
_SMALL = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
_LARGE = [1010, 1030, 1050, 1070, 1090, 1110, 1130, 1150, 1170, 1190]


def _entries(prefix, sizes, seed=1):
    return [
        ("%s-%d" % (prefix, i), _sketch("%s-%d" % (prefix, i), size, seed=seed), size)
        for i, size in enumerate(sizes)
    ]


def _query(size=50):
    """A query sketch sharing the pool tokens, so candidates are non-empty."""
    return _sketch("q", size)


class TestEnsembleMerge(unittest.TestCase):
    def _mergeable(self, entries, **kwargs):
        options = dict(threshold=0.5, num_perm=16, num_part=4, m=4, mergeable=True)
        options.update(kwargs)
        ensemble = MinHashLSHEnsemble(**options)
        if entries:
            ensemble.index(entries)
        return ensemble

    def _reference(self, entries, **kwargs):
        options = dict(threshold=0.5, num_perm=16, num_part=4, m=4)
        options.update(kwargs)
        ensemble = MinHashLSHEnsemble(**options)
        ensemble.index(entries)
        return ensemble

    def test_shards_really_have_three_nonempty_partitions_and_different_bounds(self):
        left, right = _entries("a", _SMALL), _entries("b", _LARGE)
        e1, e2 = self._mergeable(left), self._mergeable(right)
        for ensemble in (e1, e2):
            nonempty = sum(1 for upper in ensemble.uppers if upper is not None)
            self.assertGreaterEqual(nonempty, 3)
            nonempty_indexes = [i for i, upper in enumerate(ensemble.uppers) if upper is not None]
            self.assertGreaterEqual(len(nonempty_indexes), 3)
        # The shard bounds are disjoint, so neither layout can serve the
        # combined data.
        self.assertLess(e1.uppers[-1], e2.lowers[0])

    def test_merge_matches_single_shot_partitions_bands_and_candidates(self):
        left, right = _entries("a", _SMALL), _entries("b", _LARGE)
        merged = self._mergeable(left).merge(self._mergeable(right))
        reference = self._reference(left + right)

        self.assertEqual(
            [(int(l), int(u)) for l, u in zip(merged.lowers, merged.uppers)],
            [(int(l), int(u)) for l, u in zip(reference.lowers, reference.uppers)],
        )

        # Structural check: same key -> partition and same band bucket keys.
        def layout(ensemble):
            positions = {}
            for partition, index in enumerate(ensemble.indexes):
                keys = set()
                for lsh in index.values():
                    keys.update(lsh.keys)
                for key in keys:
                    positions.setdefault(key, set()).add(partition)
            return {key: sorted(partitions) for key, partitions in positions.items()}

        self.assertEqual(layout(merged), layout(reference))

        # Band selection, bucket keys and candidates for every indexed set.
        for key, minhash, size in left + right:
            self.assertIn(key, merged)
            self.assertEqual(set(merged.query(minhash, size)), set(reference.query(minhash, size)))

    def test_merge_checks_full_bucket_layout(self):
        left, right = _entries("a", _SMALL), _entries("b", _LARGE)
        merged = self._mergeable(left).merge(self._mergeable(right))
        reference = self._reference(left + right)

        def snapshot(ensemble):
            buckets = []
            for index in ensemble.indexes:
                for lsh in index.values():
                    tables = [{H: frozenset(keys) for H, keys in table._dict.items()} for table in lsh.hashtables]
                    keys = {key: tuple(Hs) for key, Hs in lsh.keys._dict.items()}
                    buckets.append((tables, keys))
            return buckets

        self.assertEqual(snapshot(merged), snapshot(reference))

    def test_query_sizes_near_old_and_new_partition_boundaries(self):
        left, right = _entries("a", _SMALL), _entries("b", _LARGE)
        merged = self._mergeable(left).merge(self._mergeable(right))
        reference = self._reference(left + right)
        # Sizes straddling the new boundaries, plus the shard-only boundaries.
        probe_sizes = sorted(
            {
                int(merged.lowers[0]),
                int(merged.uppers[0]),
                int(merged.lowers[1]),
                int(merged.uppers[1]),
                int(left[0][2]) - 1,
                int(left[-1][2]) + 1,
                int(right[0][2]) - 1,
                int(right[-1][2]) + 1,
                int(merged.uppers[-1]) + 1,
            }
        )
        query = _sketch("query", 50)
        query_hashvalues = query.hashvalues.copy()
        for size in probe_sizes:
            self.assertGreater(size, 0)
            self.assertEqual(set(merged.query(query, size)), set(reference.query(query, size)))
        self.assertTrue(np.array_equal(query.hashvalues, query_hashvalues))

    def test_three_way_merge_order_independent(self):
        shards = [_entries("a", _SMALL), _entries("b", _LARGE), _entries("c", [5, 500, 5000, 20000])]
        ensembles = [self._mergeable(shard) for shard in shards]
        merged_forward = ensembles[0].merge(ensembles[1]).merge(ensembles[2])
        merged_backward = ensembles[2].merge(ensembles[1]).merge(ensembles[0])
        reference = self._reference(shards[0] + shards[1] + shards[2])

        for merged in (merged_forward, merged_backward):
            self.assertEqual(
                [(int(l), int(u)) for l, u in zip(merged.lowers, merged.uppers)],
                [(int(l), int(u)) for l, u in zip(reference.lowers, reference.uppers)],
            )
            for _, minhash, size in shards[0] + shards[1] + shards[2]:
                self.assertEqual(set(merged.query(minhash, size)), set(reference.query(minhash, size)))

    def test_empty_shards(self):
        left = self._mergeable(_entries("a", _SMALL))
        empty = self._mergeable([])
        merged = left.merge(empty)
        reference = self._reference(_entries("a", _SMALL))
        self.assertEqual(
            [(int(l), int(u)) for l, u in zip(merged.lowers, merged.uppers)],
            [(int(l), int(u)) for l, u in zip(reference.lowers, reference.uppers)],
        )
        self.assertEqual(merged.is_empty(), reference.is_empty())

        both_empty = self._mergeable([]).merge(self._mergeable([]))
        self.assertTrue(both_empty.is_empty())
        self.assertEqual(list(both_empty.query(_sketch("q", 10), 10)), [])
        pages = list(both_empty.query_paged(_sketch("q", 10), 10).pages(4))
        self.assertTrue(all(page.keys == [] for page in pages))
        self.assertTrue(pages[-1].done)
        # The empty result is itself mergeable.
        again = both_empty.merge(left)
        for key in ("a-0", "a-9"):
            self.assertIn(key, again)

    def test_merge_result_is_mergeable_and_sources_untouched(self):
        left, right = _entries("a", _SMALL), _entries("b", _LARGE)
        e1, e2 = self._mergeable(left), self._mergeable(right)
        before_1, before_2 = self._snapshot(e1), self._snapshot(e2)
        merged = e1.merge(e2)
        self.assertEqual(self._snapshot(e1), before_1)
        self.assertEqual(self._snapshot(e2), before_2)

        third = self._mergeable(_entries("c", [7, 700, 7000]))
        twice = merged.merge(third)
        reference = self._reference(left + right + _entries("c", [7, 700, 7000]))
        self.assertEqual(
            [(int(l), int(u)) for l, u in zip(twice.lowers, twice.uppers)],
            [(int(l), int(u)) for l, u in zip(reference.lowers, reference.uppers)],
        )

    def test_result_is_independent_of_sources(self):
        left, right = _entries("a", _SMALL), _entries("b", _LARGE)
        e1, e2 = self._mergeable(left), self._mergeable(right)
        merged = e1.merge(e2)
        merged_key, query, size = left[0]
        before = set(merged.query(query, size))
        self.assertIn(merged_key, before)
        # Removing a key from the result must not change the sources.
        for index in merged.indexes:
            for lsh in index.values():
                if merged_key in lsh:
                    lsh.remove(merged_key)
        for key, _, _ in left:
            self.assertIn(key, e1)
        self.assertNotIn(merged_key, merged)
        self.assertNotIn(merged_key, set(merged.query(query, size)))

        # And removing from a source must not change the result.
        source_key = right[0][0]
        for index in e2.indexes:
            for lsh in index.values():
                if source_key in lsh:
                    lsh.remove(source_key)
        self.assertIn(source_key, merged)

    def test_source_arrays_mutated_after_index(self):
        left = _entries("a", _SMALL)
        sketches = [minhash for _, minhash, _ in left]
        e1 = self._mergeable(left)
        right = self._mergeable(_entries("b", _LARGE))
        # Mutate the caller's arrays after both builds; the merge must use the
        # snapshots taken at index time, not these arrays.
        for minhash in sketches:
            minhash.hashvalues[:] = np.uint64(0)
        merged = e1.merge(right)
        reference = self._reference(
            [(key, _sketch(key, size), size) for key, _, size in _entries("a", _SMALL)]
            + _entries("b", _LARGE)
        )
        self.assertEqual(
            [(int(l), int(u)) for l, u in zip(merged.lowers, merged.uppers)],
            [(int(l), int(u)) for l, u in zip(reference.lowers, reference.uppers)],
        )
        for key, _, size in _entries("a", _SMALL):
            fresh = _sketch(key, size)
            self.assertEqual(set(merged.query(fresh, size)), set(reference.query(fresh, size)))

    def test_identical_overlap_accepted_once(self):
        shared = _entries("s", [10, 40])
        left = shared + _entries("a", _SMALL)
        right = _entries("b", _LARGE) + [(key, _sketch(key, size), size) for key, _, size in shared]
        e1, e2 = self._mergeable(left), self._mergeable(right)
        merged = e1.merge(e2)
        # The shared keys appear once each in the partition layout. Each key
        # lives in one partition, and one entry in that partition's key map.
        occurrences = {key: 0 for key, _, _ in shared}
        for index in merged.indexes:
            for lsh in index.values():
                if next(iter(index)) != lsh.r:
                    continue
                for key in lsh.keys:
                    if key in occurrences:
                        occurrences[key] += 1
        for key, _, _ in shared:
            self.assertEqual(occurrences[key], 1)
        # The merged size histogram counts an accepted overlap once, so the
        # partitions match a reference built from de-duplicated entries.
        unique = {(key, size): (key, minhash, size) for key, minhash, size in left}
        unique.update({(key, size): (key, minhash, size) for key, minhash, size in _entries("b", _LARGE)})
        reference = self._reference(list(unique.values()))
        self.assertEqual(
            [(int(l), int(u)) for l, u in zip(merged.lowers, merged.uppers)],
            [(int(l), int(u)) for l, u in zip(reference.lowers, reference.uppers)],
        )

    def test_overlap_error_policy(self):
        shared = _entries("s", [10, 40])
        e1 = self._mergeable(shared + _entries("a", _SMALL))
        e2 = self._mergeable(_entries("b", _LARGE) + shared)
        with self.assertRaises(ValueError):
            e1.merge(e2, overlap="error")
        # identical accepts the same data.
        self.assertFalse(e1.merge(e2, overlap="identical").is_empty())

    def test_overlap_identical_matches_plain_union_of_unique_entries(self):
        shared_left = _entries("s", [10, 40])
        shared_right = _entries("s", [10, 40])
        left = self._mergeable(shared_left + _entries("a", _SMALL))
        right = self._mergeable(_entries("b", _LARGE) + shared_right)
        merged = left.merge(right)
        unique = {key: (key, minhash, size) for key, minhash, size in shared_left + _entries("a", _SMALL)}
        unique.update({key: (key, minhash, size) for key, minhash, size in _entries("b", _LARGE)})
        reference = self._reference(list(unique.values()))
        self.assertEqual(
            [(int(l), int(u)) for l, u in zip(merged.lowers, merged.uppers)],
            [(int(l), int(u)) for l, u in zip(reference.lowers, reference.uppers)],
        )
        self.assertEqual(self._layout(merged), self._layout(reference))
        for value in unique.values():
            key, minhash, size = value
            self.assertEqual(set(merged.query(minhash, size)), set(reference.query(minhash, size)))
            self.assertIn(key, merged)

    @staticmethod
    def _layout(ensemble):
        positions = {}
        for partition, index in enumerate(ensemble.indexes):
            keys = set()
            for lsh in index.values():
                keys.update(lsh.keys)
            for key in keys:
                positions.setdefault(key, set()).add(partition)
        return {key: sorted(partitions) for key, partitions in positions.items()}

    def test_overlap_conflicts_raise(self):
        key, size = "shared", 50
        sketch = _sketch(key, size)
        e1 = self._mergeable([(key, sketch, size)])
        # Different hash values, same size.
        other = _sketch(key, size)
        other.hashvalues[:] = (other.hashvalues + 1).astype(np.uint64)
        e2 = self._mergeable([(key, other, size)])
        with self.assertRaises(ValueError):
            e1.merge(e2)
        # Different size, same hash values.
        e3 = self._mergeable([(key, sketch, size + 1)])
        with self.assertRaises(ValueError):
            e1.merge(e3)
        # Sources unchanged after the failure.
        self.assertIn(key, e1)
        self.assertIn(key, e2)

    def test_late_conflict_leaves_sources_intact(self):
        # A conflicting key arrives after many non-conflicting ones, so a
        # merge that wrote as it went would have committed those already.
        conflict = _sketch("shared", 50)
        left = self._mergeable([("shared", conflict, 50)])
        changed = _sketch("shared", 50)
        changed.hashvalues[-1] = np.uint64(int(changed.hashvalues[-1]) % (1 << 31) + 1)
        right_entries = [*_entries("b", _LARGE), ("shared", changed, 50)]
        right = self._mergeable(right_entries)
        before_left, before_right = self._snapshot(left), self._snapshot(right)
        with self.assertRaises(ValueError):
            left.merge(right)
        self.assertEqual(self._snapshot(left), before_left)
        self.assertEqual(self._snapshot(right), before_right)

    def test_late_conflict_under_error_policy(self):
        shared = _entries("s", [10, 40])
        left = self._mergeable(shared)
        right = self._mergeable(_entries("b", _LARGE) + shared)
        before_left, before_right = self._snapshot(left), self._snapshot(right)
        with self.assertRaises(ValueError):
            left.merge(right, overlap="error")
        self.assertEqual(self._snapshot(left), before_left)
        self.assertEqual(self._snapshot(right), before_right)

    def test_merge_does_not_deliver_partial_result_on_allocation_error(self):
        from unittest.mock import patch as mock_patch

        left = self._mergeable(_entries("a", _SMALL))
        right = self._mergeable(_entries("b", _LARGE))
        before_left, before_right = self._snapshot(left), self._snapshot(right)
        calls = {"n": 0}

        def failing(self_, entries):
            calls["n"] += 1
            raise MemoryError("injected while building the merged index")

        with mock_patch.object(MinHashLSHEnsemble, "_index_prepared", failing), self.assertRaises(MemoryError):
            left.merge(right)
        self.assertEqual(calls["n"], 1)
        self.assertEqual(self._snapshot(left), before_left)
        self.assertEqual(self._snapshot(right), before_right)
        # The sources still answer queries while the result was never built.
        self.assertFalse(left.is_empty())
        query = _query()
        self.assertTrue(set(left.query(query, 50)))

    def test_failed_merge_leaves_cursors_usable(self):
        left = self._mergeable([("shared", _sketch("shared", 50), 50), *_entries("a", _SMALL)])
        changed = _sketch("shared", 50)
        changed.hashvalues[3] = np.uint64(int(changed.hashvalues[3]) % (1 << 20) + 5)
        right = self._mergeable([*_entries("b", _LARGE), ("shared", changed, 50)])
        query = _query()
        expected_left = set(left.query(query, 50))
        cursor = left.query_paged(query, 50)
        first = cursor.next_page(3)
        with self.assertRaises(ValueError):
            left.merge(right)
        keys = list(first.keys)
        for page in cursor.pages(8):
            keys.extend(page.keys)
        self.assertEqual(set(keys), expected_left)

    def test_result_storage_is_not_shared_with_sources(self):
        left = self._mergeable(_entries("a", _SMALL))
        right = self._mergeable(_entries("b", _LARGE))
        merged = left.merge(right)
        for duplicate, source in ((merged, left), (merged, right)):
            for index in duplicate.indexes:
                for lsh in index.values():
                    for source_index in source.indexes:
                        for source_lsh in source_index.values():
                            for table in lsh.hashtables:
                                for source_table in source_lsh.hashtables:
                                    self.assertIsNot(table._dict, source_table._dict)
                            self.assertIsNot(lsh.keys._dict, source_lsh.keys._dict)

    def test_result_storage_mutation_does_not_reach_sources(self):
        left = self._mergeable(_entries("a", _SMALL))
        right = self._mergeable(_entries("b", _LARGE))
        merged = left.merge(right)
        source_key, source_sketch, source_size = _entries("b", _LARGE)[0]
        # Delete every occurrence of the key from the result's underlying dicts.
        for index in merged.indexes:
            for lsh in index.values():
                for table in lsh.hashtables:
                    for keys in table._dict.values():
                        keys.discard(source_key)
                    for key in list(lsh.keys._dict):
                        if key == source_key:
                            del lsh.keys._dict[key]
        self.assertNotIn(source_key, merged)
        self.assertIn(source_key, right)
        self.assertIn(source_key, set(right.query(source_sketch, source_size)))

    def test_most_keys_stay_still_matches_single_shot(self):
        # Both shards draw from the same evenly-spread size distribution, so
        # their partitions line up and almost no key moves. The result must
        # still match a single-shot build exactly.
        base = list(range(10, 410, 10))
        sizes_a = [size for size in base for _ in range(5)]
        sizes_b = [size for size in base for _ in range(5)]
        left_entries = [("a-%d" % i, _sketch("a-%d" % i, size), size) for i, size in enumerate(sizes_a)]
        right_entries = [("b-%d" % i, _sketch("b-%d" % i, size), size) for i, size in enumerate(sizes_b)]
        left, right = self._mergeable(left_entries), self._mergeable(right_entries)
        before = {}
        for ensemble in (left, right):
            for partition, index in enumerate(ensemble.indexes):
                for lsh in index.values():
                    for key in lsh.keys:
                        before.setdefault(key, set()).add(partition)
        merged = left.merge(right)
        after = self._layout(merged)
        moved = sum(1 for key, parts in after.items() if sorted(before.get(key, ())) != parts)
        self.assertLess(moved, len(after) // 10)
        reference = self._reference(left_entries + right_entries)
        self.assertEqual(
            [(int(l), int(u)) for l, u in zip(merged.lowers, merged.uppers)],
            [(int(l), int(u)) for l, u in zip(reference.lowers, reference.uppers)],
        )
        self.assertEqual(self._layout(merged), self._layout(reference))

    def test_most_keys_move_still_matches_single_shot(self):
        # One shard holds small sets, the other large ones: nearly every key
        # moves to a different partition number. The result must still match.
        left_entries = _entries("a", _SMALL)
        right_entries = _entries("b", _LARGE)
        left, right = self._mergeable(left_entries), self._mergeable(right_entries)
        before = {}
        for ensemble in (left, right):
            for partition, index in enumerate(ensemble.indexes):
                for lsh in index.values():
                    for key in lsh.keys:
                        before.setdefault(key, set()).add(partition)
        merged = left.merge(right)
        after = self._layout(merged)
        moved = sum(1 for key, parts in after.items() if before.get(key) != parts)
        self.assertGreater(moved, len(after) // 2)
        reference = self._reference(left_entries + right_entries)
        self.assertEqual(self._layout(merged), self._layout(reference))

    def test_retained_state_is_compact_and_immutable(self):
        left = self._mergeable(_entries("a", _SMALL))
        # Retained hash values are immutable bytes, never views on or copies of
        # numpy arrays held by the caller.
        for entry in left._entries.values():
            self.assertIsInstance(entry.hashvalues, bytes)
            self.assertEqual(len(entry.hashvalues), 16 * 8)
        merged = left.merge(self._mergeable(_entries("b", _LARGE)))
        for entry in merged._entries.values():
            self.assertIsInstance(entry.hashvalues, bytes)
        # The merge reuses the same bytes objects rather than copying them.
        overlapped = [key for key in left._entries if key in merged._entries]
        self.assertTrue(overlapped)
        for key in overlapped:
            self.assertIs(left._entries[key].hashvalues, merged._entries[key].hashvalues)

    def test_retained_sketch_exposes_a_read_only_view(self):
        from datasketch.lshensemble import _RetainedSketch

        payload = np.arange(16, dtype=np.uint64).tobytes()
        sketch = _RetainedSketch(payload, seed=1)
        self.assertEqual(len(sketch), 16)
        self.assertEqual(len(sketch.hashvalues), 16)
        self.assertFalse(sketch.hashvalues.flags.writeable)
        # The array is a view over the retained bytes, not a fresh copy.
        self.assertTrue(np.shares_memory(sketch.hashvalues, np.frombuffer(payload, dtype=np.uint64)))

    def test_merge_does_not_materialize_a_key_by_num_perm_matrix(self):
        # The merge inserts keys from per-key immutable snapshots. A full
        # key-count x num_perm matrix would show up as a 2-D numpy allocation;
        # check none is created by watching ndarray allocations during merge.
        seen_matrices = []
        left = self._mergeable(_entries("a", _SMALL))
        right = self._mergeable(_entries("b", _LARGE))
        original = np.array

        def watching_array(obj, *args, **kwargs):
            array = original(obj, *args, **kwargs)
            if getattr(array, "ndim", 0) == 2 and array.shape[0] > 1 and array.shape[1] > 1:
                seen_matrices.append(array.shape)
            return array

        with mock.patch.object(np, "array", watching_array):
            merged = left.merge(right)
        self.assertFalse(merged.is_empty())
        # A key x num_perm matrix would have num_perm (16) columns and one row
        # per key. The size-histogram array is only 2 columns wide, so it is
        # not such a matrix.
        self.assertFalse([shape for shape in seen_matrices if shape[1] >= left.h])

    def test_pickle_round_trip_keeps_mergeability(self):
        import pickle

        left = self._mergeable(_entries("a", _SMALL))
        right = self._mergeable(_entries("b", _LARGE))
        restored = pickle.loads(pickle.dumps(left))
        merged = restored.merge(right)
        reference = self._reference(_entries("a", _SMALL) + _entries("b", _LARGE))
        self.assertEqual(
            [(int(l), int(u)) for l, u in zip(merged.lowers, merged.uppers)],
            [(int(l), int(u)) for l, u in zip(reference.lowers, reference.uppers)],
        )
    def test_key_encoding_paths(self):
        for prepickle in (False, True):
            left = _entries("a", _SMALL)
            right = _entries("b", _LARGE)
            e1 = self._mergeable(left, prepickle=prepickle)
            e2 = self._mergeable(right, prepickle=prepickle)
            merged = e1.merge(e2)
            self.assertEqual(merged._prepickle, prepickle)
            self.assertTrue(all(key in merged for key, _, _ in left + right))
            for key, minhash, size in left + right:
                self.assertIn(key, set(merged.query(minhash, size)))

    def test_key_types(self):
        converters = {
            "str": lambda index: "k%d" % index,
            "bytes": lambda index: b"k%d" % index,
            "tuple": lambda index: ("k", index),
        }
        for key_type, convert in converters.items():
            left = [(convert(i), _sketch("a%d" % i, size), size) for i, size in enumerate(_SMALL)]
            right = [(convert(i + 100), _sketch("b%d" % i, size), size) for i, size in enumerate(_LARGE)]
            merged = self._mergeable(left).merge(self._mergeable(right))
            for key, minhash, size in left + right:
                self.assertIn(key, merged)
                self.assertIn(key, set(merged.query(minhash, size)))
            self.assertGreater(len(merged._entries), 0, key_type)

    def test_mergeable_index_rejects_unsupported_sketches(self):
        generator = WeightedMinHashGenerator(100, sample_size=16)
        weighted = generator.minhash(np.random.default_rng(0).uniform(1, 10, 100))
        ensemble = MinHashLSHEnsemble(threshold=0.5, num_perm=16, num_part=4, m=4, mergeable=True)
        with self.assertRaises(TypeError):
            ensemble.index([("w", weighted, 100)])
        self.assertTrue(ensemble.is_empty())

        lean = LeanMinHash(_sketch("lean", 10))
        with self.assertRaises(TypeError):
            ensemble.index([("l", lean, 10)])
        self.assertTrue(ensemble.is_empty())

    def test_mergeable_index_rejects_mixed_coordinates(self):
        first = _sketch("a", 10)
        second = _sketch("b", 20)
        permutations = second.permutations.copy()
        permutations[0, 0] = (permutations[0, 0] % 1000) + 7
        second = MinHash(hashvalues=second.hashvalues, permutations=permutations)
        ensemble = MinHashLSHEnsemble(threshold=0.5, num_perm=16, num_part=4, m=4, mergeable=True)
        with self.assertRaises(ValueError):
            ensemble.index([("a", first, 10), ("b", second, 20)])
        self.assertTrue(ensemble.is_empty())

    def test_mergeable_index_rejects_other_seed(self):
        first = _sketch("a", 10, seed=1)
        second = _sketch("b", 20, seed=2)
        ensemble = MinHashLSHEnsemble(threshold=0.5, num_perm=16, num_part=4, m=4, mergeable=True)
        with self.assertRaises(ValueError):
            ensemble.index([("a", first, 10), ("b", second, 20)])
        self.assertTrue(ensemble.is_empty())

    def test_merge_rejects_different_configuration_even_when_empty(self):
        base = dict(threshold=0.5, num_perm=16, num_part=4, m=4, mergeable=True)
        reference = self._mergeable(_entries("a", _SMALL))
        variations = [
            dict(base, threshold=0.6),
            dict(base, num_perm=32, m=4),
            dict(base, num_part=8),
            dict(base, m=8),
            dict(base, weights=(0.4, 0.6)),
        ]
        for variation in variations:
            empty = MinHashLSHEnsemble(**variation)
            with self.assertRaises(ValueError):
                reference.merge(empty)
        # A different key encoding is also a configuration mismatch.
        encoded = MinHashLSHEnsemble(**dict(base, prepickle=True))
        with self.assertRaises(ValueError):
            reference.merge(encoded)

    def test_merge_rejects_different_coordinates(self):
        left = _sketch("a", 10, seed=1)
        other = _sketch("b", 20, seed=2)
        e1 = self._mergeable([("a", left, 10)])
        e2 = self._mergeable([("b", other, 20)])
        with self.assertRaises(ValueError):
            e1.merge(e2)

        # Same seed, different permutation arrays.
        different = _sketch("b", 20)
        permutations = different.permutations.copy()
        permutations[1, 0] = (permutations[1, 0] % 1000) + 7
        different = MinHash(hashvalues=different.hashvalues, permutations=permutations)
        e3 = self._mergeable([("c", different, 30)])
        with self.assertRaises(ValueError):
            e1.merge(e3)

    def test_merge_rejects_non_mergeable_index_before_touching_content(self):
        plain = MinHashLSHEnsemble(threshold=0.5, num_perm=16, num_part=4, m=4)
        plain.index(_entries("a", _SMALL))
        mergeable = self._mergeable(_entries("b", _LARGE))
        before = self._snapshot(plain)
        with self.assertRaises(MergeNotSupportedError):
            plain.merge(mergeable)
        with self.assertRaises(MergeNotSupportedError):
            mergeable.merge(plain)
        self.assertEqual(self._snapshot(plain), before)

    def test_merge_rejects_non_dict_storage(self):
        from unittest.mock import patch

        from test.test_lshensemble import fake_redis

        storage_config = {"type": "redis", "redis": {"host": "localhost", "port": 6379}}
        with patch("redis.Redis", fake_redis):
            redis_index = MinHashLSHEnsemble(
                threshold=0.5, num_perm=16, num_part=4, m=4, storage_config=storage_config, mergeable=True
            )
            redis_index.index(_entries("a", _SMALL))
            dict_index = self._mergeable(_entries("b", _LARGE))
            with self.assertRaises(UnsupportedStorageError):
                redis_index.merge(dict_index)
            with self.assertRaises(UnsupportedStorageError):
                dict_index.merge(redis_index)
            # The redis-backed index still queries as before.
            self.assertIn("a-0", redis_index)

    def test_merge_result_uses_paged_query(self):
        left, right = _entries("a", _SMALL), _entries("b", _LARGE)
        merged = self._mergeable(left).merge(self._mergeable(right))
        query = _sketch("q", 50)
        expected = set(merged.query(query, 50))
        cursor = merged.query_paged(query, 50)
        keys = []
        for page in cursor.pages(8):
            keys.extend(page.keys)
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(set(keys), expected)

    def test_existing_cursors_survive_merge(self):
        left, right = _entries("a", _SMALL), _entries("b", _LARGE)
        e1, e2 = self._mergeable(left), self._mergeable(right)
        query = _sketch("q", 50)
        cursor_1 = e1.query_paged(query, 50)
        cursor_2 = e2.query_paged(query, 50)
        first_1 = cursor_1.next_page(4)
        first_2 = cursor_2.next_page(4)
        expected_1 = set(e1.query(query, 50))
        expected_2 = set(e2.query(query, 50))
        merged = e1.merge(e2)
        rest_1, rest_2 = list(first_1.keys), list(first_2.keys)
        for page in cursor_1.pages(8):
            rest_1.extend(page.keys)
        for page in cursor_2.pages(8):
            rest_2.extend(page.keys)
        self.assertEqual(set(rest_1), expected_1)
        self.assertEqual(set(rest_2), expected_2)
        self.assertFalse(merged.is_empty())

    @staticmethod
    def _snapshot(ensemble):
        state = [
            tuple(None if bound is None else int(bound) for bound in ensemble.lowers),
            tuple(None if bound is None else int(bound) for bound in ensemble.uppers),
        ]
        for index in ensemble.indexes:
            for lsh in index.values():
                tables = tuple(
                    tuple(sorted((H, tuple(sorted(keys))) for H, keys in table._dict.items()))
                    for table in lsh.hashtables
                )
                keys = tuple(sorted((key, tuple(Hs)) for key, Hs in lsh.keys._dict.items()))
                state.append((tables, keys))
        return tuple(state)


if __name__ == "__main__":
    unittest.main()
