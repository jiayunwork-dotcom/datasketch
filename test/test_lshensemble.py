import pickle
import unittest
from unittest.mock import patch

import mockredis
import numpy as np

from datasketch.lshensemble import MinHashLSHEnsemble
from datasketch.minhash import MinHash
from datasketch.storage import UnsupportedStorageError
from datasketch.weighted_minhash import WeightedMinHashGenerator


def fake_redis(**kwargs):
    redis = mockredis.mock_redis_client(**kwargs)
    redis.connection_pool = None
    redis.response_callbacks = None
    return redis


def _minhash_from_hashvalues(hashvalues, num_perm):
    return MinHash(num_perm=num_perm, hashvalues=np.array(hashvalues, dtype=np.uint64))


class _TwoPartitionFixture:
    """An index with exactly two non-empty partitions whose selected bands are
    at most eight each, both hit by the query.

    Partition 0 holds one single-key bucket (the "small" key), which shares a
    band with the query but appears nowhere in the big bucket. Partition 1
    holds a bucket with 1000 keys. The small key's band in partition 0 is
    reached only after the first bands of both partitions have been probed,
    which is exactly what the paged scheduler must not postpone behind the
    1000-key bucket.
    """

    num_perm = 8
    small_key = "small"
    total_big = 1000

    def __init__(self):
        self.query = MinHash(num_perm=self.num_perm, seed=1)
        for value in range(self.num_perm):
            self.query.update(repr(value).encode())
        qhv = self.query.hashvalues

        small_hv = np.full(self.num_perm, 424242, dtype=np.uint64)
        small_hv[2:4] = qhv[2:4]
        small = (self.small_key, _minhash_from_hashvalues(small_hv, self.num_perm), 5)

        big = []
        for i in range(self.total_big):
            hv = np.full(self.num_perm, 10**6 + i, dtype=np.uint64)
            hv[0] = qhv[0]
            big.append(("big%d" % i, _minhash_from_hashvalues(hv, self.num_perm), self.total_big))
        self.big_keys = {key for key, _, _ in big}

        self.ensemble = MinHashLSHEnsemble(threshold=0.6, num_perm=self.num_perm, num_part=2, m=2)
        self.ensemble.index([*big, small])
        self.size = 5

    def selected_bands(self):
        """Return ``[(partition, b, r, [bucket sizes])]`` for the query."""
        selected = []
        for i, index in enumerate(self.ensemble.indexes):
            upper = self.ensemble.uppers[i]
            if upper is None:
                continue
            b, r = self.ensemble._get_optimal_param(upper, self.size)
            b, r = int(b), int(r)
            lsh = index[r]
            sizes = [
                lsh.hashtables[band].bucket_size(lsh._H(self.query.hashvalues[band * r : (band + 1) * r]))
                for band in range(b)
            ]
            selected.append((i, b, r, sizes))
        return selected


class TestMinHashLSHEnsemble(unittest.TestCase):
    def test_init(self):
        lsh = MinHashLSHEnsemble(threshold=0.8)
        self.assertTrue(lsh.is_empty())

    def _data(self, count):
        sizes = np.random.randint(1, 100, count)
        for key, size in enumerate(sizes):
            m = MinHash()
            for i in range(size):
                m.update(("%d" % i).encode("utf8"))
            yield (key, m, size)

    def test_index(self):
        lsh = MinHashLSHEnsemble(threshold=0.8)
        lsh.index(self._data(64))
        self.assertFalse(lsh.is_empty())
        self.assertTrue(41 in lsh)

    def test_query(self):
        lsh = MinHashLSHEnsemble(threshold=0.9)
        data = list(self._data(64))
        lsh.index(data)
        for key, minhash, size in data:
            keys = lsh.query(minhash, size)
            self.assertTrue(key in keys)

    def test_pickle(self):
        lsh = MinHashLSHEnsemble(threshold=0.9)
        data = list(self._data(32))
        lsh.index(data)
        buf = pickle.dumps(lsh)
        lsh2 = pickle.loads(buf)
        for _key, minhash, size in data:
            keys1 = lsh.query(minhash, size)
            keys2 = lsh2.query(minhash, size)
            self.assertTrue(set(keys1) == set(keys2))

    def test_index_redis(self):
        storage_config = {
            "type": "redis",
            "redis": {
                "host": "localhost",
                "port": 6379,
            },
        }
        with patch("redis.Redis", fake_redis):
            lsh = MinHashLSHEnsemble(threshold=0.8, storage_config=storage_config)
            lsh.index(self._data(64))
            self.assertFalse(lsh.is_empty())
            self.assertTrue(41 in lsh)

    def test_query_redis(self):
        storage_config = {
            "type": "redis",
            "redis": {
                "host": "localhost",
                "port": 6379,
            },
        }
        with patch("redis.Redis", fake_redis):
            lsh = MinHashLSHEnsemble(threshold=0.9, storage_config=storage_config)
            data = list(self._data(64))
            lsh.index(data)
            for key, minhash, size in data:
                keys = lsh.query(minhash, size)
                self.assertTrue(key in keys)


if __name__ == "__main__":
    unittest.main()


class TestMinHashLSHEnsemblePagedQuery(unittest.TestCase):
    def setUp(self):
        self.fixture = _TwoPartitionFixture()

    def test_fixture_has_two_partitions_with_at_most_eight_bands(self):
        selected = self.fixture.selected_bands()
        self.assertEqual(len(selected), 2)
        for _, b, _, _ in selected:
            self.assertLessEqual(b, 8)
        # Both partitions are hit by the query.
        self.assertTrue(any(1 in sizes for _, _, _, sizes in selected))
        self.assertTrue(any(self.fixture.total_big in sizes for _, _, _, sizes in selected))

    def test_small_partition_key_within_two_pages(self):
        query = self.fixture.ensemble.query_paged(self.fixture.query, self.fixture.size)
        seen = []
        for _ in range(2):
            page = query.next_page(16)
            seen.extend(page.keys)
        self.assertIn(self.fixture.small_key, seen)

    def test_pages_agree_with_original_query(self):
        expected = set(self.fixture.ensemble.query(self.fixture.query, self.fixture.size))
        query = self.fixture.ensemble.query_paged(self.fixture.query, self.fixture.size)
        keys = []
        pages = 0
        while True:
            page = query.next_page(16)
            pages += 1
            keys.extend(page.keys)
            if page.done:
                break
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(set(keys), expected)
        self.assertFalse(query.closed)

    def test_budget_one_and_repeated_hits(self):
        # One unit per page: lookups and reads are charged one at a time, and
        # pages beyond the first still make progress.
        query = self.fixture.ensemble.query_paged(self.fixture.query, self.fixture.size)
        first = query.next_page(1)
        self.assertEqual(first.budget_used, 1)
        self.assertEqual(first.keys, [])
        self.assertFalse(first.done)
        second = query.next_page(1)
        self.assertEqual(second.budget_used, 1)

    def test_duplicate_bucket_hits_are_charged(self):
        # Make every band of the query collide with the same single key, so
        # the key is read once per band but delivered once.
        ensemble = MinHashLSHEnsemble(threshold=0.8, num_perm=8, num_part=2, m=2)
        m = MinHash(num_perm=8)
        m.update(b"same")
        ensemble.index([("dup", m.copy(), 10)])
        expected_bands = sum(
            int(ensemble._get_optimal_param(upper, 10)[0]) for upper in ensemble.uppers if upper is not None
        )
        query = ensemble.query_paged(m, 10)
        keys = []
        spent = 0
        while True:
            page = query.next_page(4)
            spent += page.budget_used
            keys.extend(page.keys)
            if page.done:
                break
        self.assertEqual(keys.count("dup"), 1)
        # Every band is counted by an explicit lookup, then the key is read
        # from each of the colliding bands.
        self.assertGreaterEqual(spent, expected_bands + 1)
        self.assertEqual(query.budget_used, spent)

    def test_metering_tracks_bucket_access_not_page_keys(self):
        # Wrap the storage accessors and check each unit maps to one lookup or
        # one key read, and that no bucket is materialized wholesale.
        fixture = self.fixture
        lookups = []
        reads = []
        for index in fixture.ensemble.indexes:
            for lsh in index.values():
                original_lookup = lsh._paged_bucket_lookup

                def counting_lookup(hashvalues, band, _original=original_lookup):
                    result = _original(hashvalues, band)
                    lookups.append(result)
                    return result

                lsh._paged_bucket_lookup = counting_lookup
                for table in lsh.hashtables:
                    original_cursor = table.bucket_cursor

                    def counting_cursor(key, _original=original_cursor):
                        cursor = _original(key)

                        def wrapped():
                            for value in cursor:
                                reads.append(key)
                                yield value

                        return wrapped()

                    table.bucket_cursor = counting_cursor
        query = fixture.ensemble.query_paged(fixture.query, fixture.size)
        spent = 0
        while not query.done:
            spent += query.next_page(16).budget_used
        self.assertEqual(spent, len(lookups) + len(reads))
        self.assertGreater(len(lookups), 0)
        self.assertGreater(len(reads), 0)
        # Every key read is counted, including repeats of a delivered key.
        self.assertGreaterEqual(len(reads), fixture.total_big)

    def test_empty_page_on_empty_buckets(self):
        # With one in-memory set, a budget of one pays for the single bucket
        # lookup, which is empty for the query. The page is empty yet the query
        # is not done, because the partition's other bands are still unread.
        ensemble = MinHashLSHEnsemble(threshold=0.9, num_perm=8, num_part=2, m=2)
        m = MinHash(num_perm=8)
        m.update(b"a")
        other = MinHash(num_perm=8)
        other.update(b"b")
        ensemble.index([("a", m, 10), ("b", other, 10)])
        query = ensemble.query_paged(m, 10)
        page = query.next_page(1)
        self.assertEqual(page.keys, [])
        self.assertEqual(page.budget_used, 1)
        self.assertFalse(page.done)

    def test_pages_helper(self):
        query = self.fixture.ensemble.query_paged(self.fixture.query, self.fixture.size)
        keys = []
        budgets = []
        for page in query.pages(16):
            keys.extend(page.keys)
            budgets.append(page.budget_used)
        self.assertTrue(budgets)
        self.assertTrue(all(b <= 16 for b in budgets))
        self.assertEqual(set(keys), set(self.fixture.ensemble.query(self.fixture.query, self.fixture.size)))

    def test_weighted_minhash(self):
        mg = WeightedMinHashGenerator(100, sample_size=32)
        ensemble = MinHashLSHEnsemble(threshold=0.5, num_perm=32, num_part=2, m=4)
        rng = np.random.default_rng(0)
        ensemble.index([("w%d" % i, mg.minhash(rng.uniform(1, 10, 100)), 100) for i in range(20)])
        query_minhash = mg.minhash(rng.uniform(1, 10, 100))
        expected = set(ensemble.query(query_minhash, 100))
        query = ensemble.query_paged(query_minhash, 100)
        keys = []
        for page in query.pages(4):
            keys.extend(page.keys)
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(set(keys), expected)

    def test_interleaved_queries(self):
        fixture = self.fixture
        q1 = fixture.ensemble.query_paged(fixture.query, fixture.size)
        q2 = fixture.ensemble.query_paged(fixture.query, fixture.size)
        keys1, keys2 = [], []
        for _ in range(3):
            keys1.extend(q1.next_page(7).keys)
            keys2.extend(q2.next_page(5).keys)
        while not q1.done:
            keys1.extend(q1.next_page(7).keys)
        while not q2.done:
            keys2.extend(q2.next_page(5).keys)
        expected = set(fixture.ensemble.query(fixture.query, fixture.size))
        self.assertEqual(set(keys1), expected)
        self.assertEqual(set(keys2), expected)

    def test_empty_page_over_empty_buckets_keeps_going(self):
        # A page whose budget only covers lookups of empty buckets has no new
        # keys but is not final, because bucket lookups are charged as well and
        # a later partition still has a live bucket.
        fixture = _TwoPartitionFixture()
        query = fixture.ensemble.query_paged(fixture.query, fixture.size)
        page = query.next_page(1)
        self.assertEqual(page.budget_used, 1)
        self.assertFalse(page.done)
        remaining = []
        while not query.done:
            next_page = query.next_page(16)
            remaining.extend(next_page.keys)
        self.assertEqual(set(page.keys + remaining), set(fixture.ensemble.query(fixture.query, fixture.size)))

    def test_invalid_budget_does_not_advance(self):
        fixture = self.fixture
        query = fixture.ensemble.query_paged(fixture.query, fixture.size)
        first = query.next_page(3)
        for bad in (0, -1, 1.5, "3", True, None):
            with self.assertRaises((ValueError, TypeError)):
                query.next_page(bad)
        self.assertEqual(query.budget_used, first.budget_used)
        second = query.next_page(3)
        self.assertEqual(second.budget_used, 3)

    def test_query_snapshot_and_close(self):
        fixture = self.fixture
        query = fixture.ensemble.query_paged(fixture.query, fixture.size)
        snapshot = query._hashvalues.copy()
        fixture.query.update(b"mutated")
        self.assertTrue(np.array_equal(query._hashvalues, snapshot))
        query = fixture.ensemble.query_paged(fixture.query, fixture.size)
        query.next_page(16)
        query.close()
        with self.assertRaises(RuntimeError):
            query.next_page(3)
        query.close()  # closing twice is fine
        # The context manager closes on exit as well.
        with fixture.ensemble.query_paged(fixture.query, fixture.size) as scoped:
            scoped.next_page(4)
        with self.assertRaises(RuntimeError):
            scoped.next_page(4)

    def test_unsupported_storage_rejected_before_bucket_access(self):
        storage_config = {"type": "redis", "redis": {"host": "localhost", "port": 6379}}
        with patch("redis.Redis", fake_redis):
            ensemble = MinHashLSHEnsemble(threshold=0.6, num_perm=8, num_part=2, m=2, storage_config=storage_config)
            m = MinHash(num_perm=8)
            m.update(b"redis-key")
            ensemble.index([("redis-key", m, 10)])
            with self.assertRaises(UnsupportedStorageError):
                ensemble.query_paged(m, 10)
            # Old query path still works on Redis-backed indexes.
            self.assertEqual(len(list(ensemble.query(m, 10))), 1)

    def test_original_query_and_contains_unchanged(self):
        fixture = self.fixture
        expected = set(fixture.ensemble.query(fixture.query, fixture.size))
        self.assertEqual(len(expected), fixture.total_big + 1)
        self.assertIn(fixture.small_key, fixture.ensemble)
        self.assertIn("big0", fixture.ensemble)
        self.assertNotIn("missing", fixture.ensemble)

    def test_pickle_still_works(self):
        fixture = self.fixture
        restored = pickle.loads(pickle.dumps(fixture.ensemble))
        self.assertEqual(
            set(restored.query(fixture.query, fixture.size)),
            set(fixture.ensemble.query(fixture.query, fixture.size)),
        )


if __name__ == "__main__":
    unittest.main()
