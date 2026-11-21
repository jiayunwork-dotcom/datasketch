"""Baseline reproduction of the buffered maintenance-chain bug.

MinHash num_perm=8, LSH params=(2,4).  In one insertion_session, insert key
"k" with all-1 hashvalues, then again with all-2 hashvalues.

Expected buggy behaviour (pre-fix):
  * Redis path (fakeredis): both inserts accepted; after the session closes,
    both queries return "k"; after remove("k"), membership is False but the
    all-2 query still returns "k".
  * Dict path: second insert raises ValueError.
"""

import pickle
import sys

import numpy as np

from datasketch.lsh import MinHashLSH
from datasketch.minhash import MinHash


def make_minhashes():
    m1 = MinHash(num_perm=8, hashvalues=np.ones(8, dtype=np.uint64))
    m2 = MinHash(num_perm=8, hashvalues=np.full(8, 2, dtype=np.uint64))
    return m1, m2


def run_dict():
    print("=== dict storage ===")
    m1, m2 = make_minhashes()
    lsh = MinHashLSH(num_perm=8, params=(2, 4))
    with lsh.insertion_session() as session:
        session.insert("k", m1)
        try:
            session.insert("k", m2)
            print("second insert: ACCEPTED")
        except ValueError as e:
            print(f"second insert: ValueError: {e}")
    print("query(all-1):", lsh.query(m1))
    print("query(all-2):", lsh.query(m2))
    try:
        lsh.remove("k")
        print("removed k; 'k' in lsh:", "k" in lsh)
    except ValueError as e:
        print(f"remove: ValueError: {e}")
    print("after remove, query(all-1):", lsh.query(m1))
    print("after remove, query(all-2):", lsh.query(m2))
    print("get_counts:", lsh.get_counts())


def run_redis():
    print("=== redis storage (fakeredis) ===")
    import fakeredis
    from unittest.mock import patch

    m1, m2 = make_minhashes()
    with patch("redis.Redis", fakeredis.FakeRedis):
        lsh = MinHashLSH(
            num_perm=8,
            params=(2, 4),
            storage_config={"type": "redis", "redis": {"host": "localhost", "port": 6379}},
        )
        with lsh.insertion_session() as session:
            session.insert("k", m1)
            try:
                session.insert("k", m2)
                print("second insert: ACCEPTED")
            except ValueError as e:
                print(f"second insert: ValueError: {e}")
        print("query(all-1):", lsh.query(m1))
        print("query(all-2):", lsh.query(m2))
        try:
            lsh.remove("k")
            print("removed k; 'k' in lsh:", "k" in lsh)
        except ValueError as e:
            print(f"remove: ValueError: {e}")
        print("after remove, query(all-1):", lsh.query(m1))
        print("after remove, query(all-2):", lsh.query(m2))
        print("get_counts:", lsh.get_counts())
        # raw key-store content for k
        raw = lsh.keys.get(pickle.dumps("k"))
        print("keys store entry for k (num Hs):", len(raw))


if __name__ == "__main__":
    run_dict()
    print()
    run_redis()
    sys.exit(0)
