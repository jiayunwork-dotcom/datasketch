"""Baseline probe: deletion session leaves zero-member bucket entries (Redis)."""

import numpy as np
import fakeredis
from unittest.mock import patch

from datasketch.lsh import MinHashLSH
from datasketch.minhash import MinHash


def main():
    with patch("redis.Redis", fakeredis.FakeRedis):
        lsh = MinHashLSH(
            num_perm=8,
            params=(2, 4),
            storage_config={"type": "redis", "redis": {"host": "localhost", "port": 6379}},
        )
        ma = MinHash(num_perm=8, hashvalues=np.ones(8, dtype=np.uint64))
        mb = MinHash(num_perm=8, hashvalues=np.full(8, 2, dtype=np.uint64))
        mc = MinHash(num_perm=8, hashvalues=np.full(8, 3, dtype=np.uint64))
        with lsh.insertion_session() as session:
            session.insert("a", ma)
            session.insert("b", mb)
            session.insert("c", mc)
        print("counts after insert:", lsh.get_counts())

        with lsh.deletion_session() as session:
            session.remove("a")
            session.remove("b")
        print("'a' in lsh:", "a" in lsh, "| 'b' in lsh:", "b" in lsh, "| 'c' in lsh:", "c" in lsh)
        print("counts after deletion session:", lsh.get_counts())
        zeros = [sum(1 for v in t.values() if v == 0) for t in lsh.get_counts()]
        print("zero-member bucket entries per band:", zeros)
        print("query(all-1):", lsh.query(ma), "query(all-3):", lsh.query(mc))

        # double remove inside one deletion session
        with lsh.deletion_session() as session:
            session.remove("c")
            try:
                session.remove("c")
                print("second remove of c: ACCEPTED (bug)")
            except ValueError as e:
                print(f"second remove of c: ValueError: {e}")
        print("counts after removing c:", lsh.get_counts())
        print("is_empty:", lsh.is_empty())


if __name__ == "__main__":
    main()
