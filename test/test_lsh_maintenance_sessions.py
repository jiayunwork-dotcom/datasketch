"""Tests for the buffered maintenance chain of :class:`MinHashLSH`.

A logical maintenance operation (insert or remove of one key) spans several
independent write buffers: the key-to-signature storage and one member-bucket
storage per band, each of which may auto-flush at different moments. These
tests pin down the contract after the fix:

* with the default ``check_duplication=True``, batch insertion rejects keys
  that are already stored *or* accepted earlier in the same session, whatever
  signature they carry, and whatever the buffer size is;
* batch removal decides existence by call order: removing a key that is not
  stored - or was already removed in the same session - raises ``ValueError``;
* a rejected operation leaves no partial change and the session stays usable;
* after a session is closed, the key storage, the per-band buckets, the query
  results and the bucket statistics all agree; emptied buckets leave no
  zero-member entries behind;
* explicit ``close()`` and exiting ``with`` both finalize a session, repeated
  ``close()`` is a no-op, and a closed session rejects new operations;
* a new index object built with the same ``basename`` reads the result;
* backend failures at close propagate to the caller.

The suite runs on the dict storage, on fakeredis (both ``prepickle`` modes),
and on a real ``redis-server`` when one is available (the fixture starts its
own throwaway server; tests only use namespaces they create).
"""

import os
import pickle
import shutil
import socket
import subprocess
import tempfile
import time
import uuid
from collections import defaultdict

import numpy as np
import pytest

from datasketch.lsh import MinHashLSH
from datasketch.minhash import MinHash

try:
    import redis as redis_module
except ImportError:  # pragma: no cover - redis extra not installed
    redis_module = None

try:
    import fakeredis
except ImportError:  # pragma: no cover
    fakeredis = None

NUM_PERM = 8
PARAMS = (2, 4)
BAND_0 = slice(0, 4)
BAND_1 = slice(4, 8)


def minhash_of(values):
    """Build a MinHash of length NUM_PERM from explicit hash values."""
    return MinHash(num_perm=NUM_PERM, hashvalues=np.array(values, dtype=np.uint64))


def bulk_minhash(i):
    """A deterministic MinHash for bulk loads; keys share band-0 buckets in
    groups of about 1000 and get private band-1 buckets.
    """
    return minhash_of([i % 1000] * 4 + [i] * 4)


# A few handcrafted signatures. A and B share their band-0 bucket only.
HV_A = [1] * 4 + [5] * 4
HV_B = [1] * 4 + [6] * 4
HV_C = [9] * 8
HV_A2 = [2] * 4 + [2] * 4  # same key, different signature (the repro)
HV_FRESH = [7] * 4 + [7] * 4


class IndexModel:
    """Independent key -> signature expectation table maintained by the test."""

    def __init__(self, lsh):
        self.lsh = lsh
        self.hashvalues = {}

    def add(self, key, values):
        self.hashvalues[key] = list(values)

    def remove(self, key):
        del self.hashvalues[key]

    def signatures(self, key):
        hv = self.hashvalues[key]
        return [self.lsh._H(np.array(hv, dtype=np.uint64)[start:end]) for start, end in self.lsh.hashranges]

    def buckets(self, band):
        result = defaultdict(set)
        for key in self.hashvalues:
            result[self.signatures(key)[band]].add(key)
        return dict(result)

    def candidates(self, values):
        probe = MinHash(num_perm=NUM_PERM, hashvalues=np.array(values, dtype=np.uint64))
        Hs = [self.lsh._H(probe.hashvalues[start:end]) for start, end in self.lsh.hashranges]
        return {key for key in self.hashvalues if any(self.signatures(key)[i] == Hs[i] for i in range(len(Hs)))}


def stored_key(lsh, key):
    return pickle.dumps(key) if lsh.prepickle else key


def assert_consistent(lsh, model, absent_keys=(), query_probes=()):
    """Assert that the key storage, the band buckets, the membership checks,
    the query results and the bucket statistics all agree with the model.
    """
    # Membership checks, both ways.
    for key in model.hashvalues:
        assert key in lsh, f"{key!r} should be a member"
    for key in absent_keys:
        assert key not in lsh, f"{key!r} should not be a member"

    # Forward storage: every key maps to exactly one complete signature set.
    assert set(lsh.keys.keys()) == {stored_key(lsh, key) for key in model.hashvalues}
    for key in model.hashvalues:
        assert list(lsh.keys.get(stored_key(lsh, key))) == model.signatures(key)
    for key in absent_keys:
        assert list(lsh.keys.get(stored_key(lsh, key))) == []

    # Reverse storage: buckets correspond to the signatures item by item,
    # and no bucket directory entry has zero members.
    counts = lsh.get_counts()
    assert len(counts) == lsh.b
    for band in range(lsh.b):
        expected_buckets = model.buckets(band)
        assert set(lsh.hashtables[band].keys()) == set(expected_buckets)
        for H, members in expected_buckets.items():
            assert lsh.hashtables[band].get(H) == {stored_key(lsh, key) for key in members}
        assert counts[band] == {H: len(members) for H, members in expected_buckets.items()}
        assert all(size > 0 for size in counts[band].values())

    # Query candidate sets.
    for values in query_probes:
        got = set(lsh.query(minhash_of(values)))
        assert got == model.candidates(values), f"query candidates mismatch for {values}"


# ---------------------------------------------------------------------------
# Storage backends
# ---------------------------------------------------------------------------


def _free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@pytest.fixture(scope="session")
def real_redis_server():
    """Start a throwaway redis-server for integration tests.

    The binary is taken from $REDIS_SERVER_BIN or PATH; if none is available
    the real-redis parametrizations are skipped.
    """
    binary = os.environ.get("REDIS_SERVER_BIN") or shutil.which("redis-server")
    if binary is None:
        pytest.skip("redis-server binary not available")
    port = _free_port()
    workdir = tempfile.mkdtemp(prefix="datasketch-test-redis-")
    proc = subprocess.Popen(  # noqa: S603 - the binary is a redis-server the test itself located
        [
            binary,
            "--port",
            str(port),
            "--bind",
            "127.0.0.1",
            "--dir",
            workdir,
            "--save",
            "",
            "--appendonly",
            "no",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 10
        while True:
            try:
                import redis

                if redis.Redis(host="127.0.0.1", port=port).ping():
                    break
            except Exception:
                if time.time() > deadline:
                    proc.kill()
                    pytest.skip("redis-server did not come up")
                time.sleep(0.05)
        yield {"host": "127.0.0.1", "port": port}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(workdir, ignore_errors=True)


class _Backend:
    """Everything a test needs to build LSH objects on one storage kind."""

    def __init__(self, name, prepickle, config_factory, server=None):
        self.name = name
        self.prepickle = prepickle
        self._config_factory = config_factory
        self.server = server  # fakeredis server, when applicable

    def storage_config(self, basename):
        return self._config_factory(basename)

    def new_lsh(self, basename=None, **kwargs):
        config = self.storage_config(basename)
        kwargs.setdefault("num_perm", NUM_PERM)
        kwargs.setdefault("params", PARAMS)
        if self.name != "dict":
            kwargs.setdefault("prepickle", self.prepickle)
        return MinHashLSH(storage_config=config, **kwargs)


def _basename():
    return b"testns_" + uuid.uuid4().hex[:12].encode("ascii")


def backend_params():
    return [
        pytest.param("dict", id="dict"),
        pytest.param("fakeredis", id="fakeredis-prepickle"),
        pytest.param("fakeredis_nopickle", id="fakeredis-bytes"),
        pytest.param("realredis", id="realredis-prepickle", marks=pytest.mark.integration),
        pytest.param("realredis_nopickle", id="realredis-bytes", marks=pytest.mark.integration),
    ]


@pytest.fixture(params=backend_params())
def backend(request, monkeypatch):
    name = request.param
    if name == "dict":
        return _Backend(name, False, lambda basename: {"type": "dict"})
    if redis_module is None:
        pytest.skip("redis package not installed")
    if name.startswith("fakeredis"):
        if fakeredis is None:
            pytest.skip("fakeredis package not installed")
        server = fakeredis.FakeServer()
        monkeypatch.setattr(redis_module, "Redis", lambda **kw: fakeredis.FakeRedis(server=server, **kw))
        prepickle = name == "fakeredis"
        return _Backend(
            name,
            prepickle,
            lambda basename: {
                "type": "redis",
                "basename": basename or _basename(),
                "redis": {"host": "localhost", "port": 6399},
            },
            server=server,
        )
    # Real redis against the throwaway server started by the fixture.
    server_params = request.getfixturevalue("real_redis_server")
    prepickle = name == "realredis"
    return _Backend(
        name,
        prepickle,
        lambda basename: {
            "type": "redis",
            "basename": basename or _basename(),
            "redis": dict(server_params),
        },
    )


# ---------------------------------------------------------------------------
# The reproduction from the bug report
# ---------------------------------------------------------------------------


class TestNearDuplicateRepro:
    def test_duplicate_insert_same_session_rejected(self, backend):
        """The original repro: same key inserted twice in one session with
        different signatures must be rejected on both storages.
        """
        lsh = backend.new_lsh()
        m1, m2 = minhash_of([1] * 8), minhash_of([2] * 8)
        with lsh.insertion_session() as session:
            session.insert(b"k", m1)
            with pytest.raises(ValueError, match="already exists"):
                session.insert(b"k", m2)
        assert set(lsh.query(m1)) == {b"k"}
        assert lsh.query(m2) == []
        lsh.remove(b"k")
        assert b"k" not in lsh
        assert lsh.query(m1) == []
        assert lsh.query(m2) == []
        assert lsh.get_counts() == [{}, {}]

    def test_duplicate_insert_same_signature_rejected(self, backend):
        lsh = backend.new_lsh()
        m1 = minhash_of([1] * 8)
        with lsh.insertion_session() as session:
            session.insert(b"k", m1)
            with pytest.raises(ValueError, match="already exists"):
                session.insert(b"k", m1)
        assert set(lsh.query(m1)) == {b"k"}
        model = IndexModel(lsh)
        model.add(b"k", [1] * 8)
        assert_consistent(lsh, model, absent_keys=[b"other"], query_probes=[[1] * 8, [2] * 8])

    def test_remove_then_no_ghost_hits(self, backend):
        """After remove, neither the old nor the would-be-duplicate signature
        may produce a hit.
        """
        lsh = backend.new_lsh()
        m1, m2 = minhash_of([1] * 8), minhash_of([2] * 8)
        with lsh.insertion_session() as session:
            session.insert(b"k", m1)
        lsh.remove(b"k")
        assert b"k" not in lsh
        assert lsh.query(m1) == []
        assert lsh.query(m2) == []
        for counts in lsh.get_counts():
            assert counts == {}


# ---------------------------------------------------------------------------
# Insertion sessions
# ---------------------------------------------------------------------------


class TestInsertionSession:
    @pytest.mark.parametrize("buffer_size", [1, 2, 3, 64, 4096])
    def test_acceptance_independent_of_buffer_size(self, backend, buffer_size):
        """The same op script must give the same accept/reject decisions and
        the same final state for any buffer size, including sizes that force
        mid-session automatic flushes.
        """
        lsh = backend.new_lsh()
        events = []

        def try_insert(session, key, values):
            try:
                session.insert(key, minhash_of(values))
                events.append(("insert-ok", key))
            except ValueError:
                events.append(("insert-rejected", key))

        with lsh.insertion_session(buffer_size=buffer_size) as session:
            try_insert(session, b"k1", [11] * 8)
            try_insert(session, b"k2", [12] * 8)
            try_insert(session, b"k1", [11] * 8)  # duplicate, same signature
            try_insert(session, b"k2", [99] * 8)  # duplicate, other signature
            try_insert(session, b"k3", [13] * 8)
            try_insert(session, b"k1", [77] * 8)  # duplicate again
        # Session stayed usable after the rejections.
        assert events == [
            ("insert-ok", b"k1"),
            ("insert-ok", b"k2"),
            ("insert-rejected", b"k1"),
            ("insert-rejected", b"k2"),
            ("insert-ok", b"k3"),
            ("insert-rejected", b"k1"),
        ]
        model = IndexModel(lsh)
        model.add(b"k1", [11] * 8)
        model.add(b"k2", [12] * 8)
        model.add(b"k3", [13] * 8)
        assert_consistent(
            lsh,
            model,
            absent_keys=[b"k4"],
            query_probes=[[11] * 8, [12] * 8, [13] * 8, [99] * 8, [77] * 8],
        )

    def test_pending_and_flushed_keys_both_rejected(self, backend):
        """A duplicate is rejected whether the first insert is still buffered
        or was already flushed by an automatic flush.
        """
        if backend.name == "dict":
            pytest.skip("dict storage has no write buffer to inspect")
        lsh = backend.new_lsh()
        with lsh.insertion_session(buffer_size=2) as session:
            session.insert(b"early", minhash_of([21] * 8))
            session.insert(b"spacer", minhash_of([20] * 8))  # flushes "early"
            session.insert(b"late", minhash_of([22] * 8))  # flushes "spacer"
            # "early" made it to the storage, "late" is still buffered.
            assert stored_key(lsh, b"early") not in lsh.keys._pending_inserts
            assert stored_key(lsh, b"late") in lsh.keys._pending_inserts
            assert lsh.keys._redis.hexists(lsh.keys._name, stored_key(lsh, b"early"))
            assert not lsh.keys._redis.hexists(lsh.keys._name, stored_key(lsh, b"late"))
            with pytest.raises(ValueError):
                session.insert(b"early", minhash_of([88] * 8))  # flushed key
            with pytest.raises(ValueError):
                session.insert(b"late", minhash_of([89] * 8))  # pending key
        assert set(lsh.query(minhash_of([21] * 8))) == {b"early"}
        assert set(lsh.query(minhash_of([22] * 8))) == {b"late"}
        assert lsh.query(minhash_of([88] * 8)) == []
        assert lsh.query(minhash_of([89] * 8)) == []

    def test_membership_check_sees_pending_insert(self, backend):
        """Existence checks account for accepted-but-unflushed inserts, while
        query visibility during a session is unchanged (flushed state only
        for the redis storage; the dict storage applies writes at once).
        """
        lsh = backend.new_lsh()
        with lsh.insertion_session(buffer_size=4096) as session:
            session.insert(b"k", minhash_of([31] * 8))
            assert b"k" in lsh  # visible to membership checks immediately
            if backend.name == "dict":
                assert set(lsh.query(minhash_of([31] * 8))) == {b"k"}
            else:
                assert lsh.query(minhash_of([31] * 8)) == []
        assert b"k" in lsh
        assert set(lsh.query(minhash_of([31] * 8))) == {b"k"}

    def test_rejection_leaves_no_partial_change(self, backend):
        if backend.name == "dict":
            pytest.skip("buffer inspection needs the redis storage")
        lsh = backend.new_lsh()
        with lsh.insertion_session(buffer_size=4096) as session:
            session.insert(b"k", minhash_of([41] * 8))
            stacks_before = [len(s._buffer.command_stack) for s in (lsh.keys, *lsh.hashtables)]
            with pytest.raises(ValueError):
                session.insert(b"k", minhash_of([42] * 8))
            stacks_after = [len(s._buffer.command_stack) for s in (lsh.keys, *lsh.hashtables)]
            assert stacks_before == stacks_after  # nothing was queued
        model = IndexModel(lsh)
        model.add(b"k", [41] * 8)
        assert_consistent(lsh, model, query_probes=[[41] * 8, [42] * 8])

    def test_check_duplication_false_keeps_old_semantics(self, backend):
        """check_duplication=False skips the existence check as before."""
        lsh = backend.new_lsh()
        with lsh.insertion_session() as session:
            session.insert(b"k", minhash_of([51] * 8), check_duplication=False)
            session.insert(b"k", minhash_of([51] * 8), check_duplication=False)
        assert b"k" in lsh
        assert set(lsh.query(minhash_of([51] * 8))) == {b"k"}


# ---------------------------------------------------------------------------
# Deletion sessions
# ---------------------------------------------------------------------------


class TestDeletionSession:
    @pytest.mark.parametrize("buffer_size", [1, 2, 3, 64, 4096])
    def test_remove_acceptance_independent_of_buffer_size(self, backend, buffer_size):
        lsh = backend.new_lsh()
        for key, values in [(b"a", HV_A), (b"b", HV_B), (b"c", HV_C)]:
            lsh.insert(key, minhash_of(values))
        events = []

        def try_remove(session, key):
            try:
                session.remove(key)
                events.append(("remove-ok", key))
            except ValueError:
                events.append(("remove-rejected", key))

        with lsh.deletion_session(buffer_size=buffer_size) as session:
            try_remove(session, b"a")
            try_remove(session, b"a")  # already removed in this session
            try_remove(session, b"missing")  # never existed
            try_remove(session, b"b")  # session still usable after errors
        assert events == [
            ("remove-ok", b"a"),
            ("remove-rejected", b"a"),
            ("remove-rejected", b"missing"),
            ("remove-ok", b"b"),
        ]
        model = IndexModel(lsh)
        model.add(b"c", HV_C)
        assert_consistent(lsh, model, absent_keys=[b"a", b"b", b"missing"], query_probes=[HV_A, HV_B, HV_C])

    def test_shared_bucket_survives_sibling_removal(self, backend):
        """A and B share the band-0 bucket; removing A must not harm B, and
        removing B afterwards must drop the bucket directory entry.
        """
        lsh = backend.new_lsh()
        with lsh.insertion_session(buffer_size=4) as session:
            session.insert(b"a", minhash_of(HV_A))
            session.insert(b"b", minhash_of(HV_B))
            session.insert(b"c", minhash_of(HV_C))
        shared_H = lsh._H(np.array(HV_A[:4], dtype=np.uint64))

        with lsh.deletion_session(buffer_size=4) as session:
            session.remove(b"a")
        assert b"b" in lsh
        assert lsh.hashtables[0].get(shared_H) == {stored_key(lsh, b"b")}
        assert set(lsh.query(minhash_of(HV_B))) == {b"b"}
        # A query with A's signature still collides with B in band 0.
        assert set(lsh.query(minhash_of(HV_A))) == {b"b"}

        with lsh.deletion_session(buffer_size=4) as session:
            session.remove(b"b")
        # The last member is gone: no zero-member entry may remain.
        assert shared_H not in set(lsh.hashtables[0].keys())
        assert lsh.get_counts()[0] == {lsh._H(np.array(HV_C[:4], dtype=np.uint64)): 1}
        assert all(size > 0 for counts in lsh.get_counts() for size in counts.values())
        model = IndexModel(lsh)
        model.add(b"c", HV_C)
        assert_consistent(lsh, model, absent_keys=[b"a", b"b"], query_probes=[HV_A, HV_B, HV_C])

    def test_last_member_removal_empties_index(self, backend):
        lsh = backend.new_lsh()
        with lsh.insertion_session(buffer_size=2) as session:
            for i, values in enumerate([HV_A, HV_B, HV_C]):
                session.insert(f"k{i}".encode(), minhash_of(values))
        with lsh.deletion_session(buffer_size=2) as session:
            for i in range(3):
                session.remove(f"k{i}".encode())
        assert lsh.is_empty()
        assert lsh.get_counts() == [{}, {}]
        assert set(lsh.keys.keys()) == set()

    def test_double_remove_flushed_and_pending(self, backend):
        """A second remove is rejected whether the first one was flushed by
        an automatic flush or is still buffered.
        """
        lsh = backend.new_lsh()
        for key, values in [(b"x", HV_A), (b"y", HV_B), (b"z", HV_C)]:
            lsh.insert(key, minhash_of(values))
        with lsh.deletion_session(buffer_size=2) as session:
            session.remove(b"x")  # flushed almost immediately (tiny buffer)
            session.remove(b"y")
            session.remove(b"z")  # likely still buffered
            for key in (b"x", b"y", b"z"):
                with pytest.raises(ValueError, match="does not exist"):
                    session.remove(key)
        assert lsh.is_empty()


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


class TestSessionLifecycle:
    def test_empty_sessions(self, backend):
        lsh = backend.new_lsh()
        lsh.insert(b"a", minhash_of(HV_A))
        with lsh.insertion_session():
            pass
        with lsh.deletion_session():
            pass
        model = IndexModel(lsh)
        model.add(b"a", HV_A)
        assert_consistent(lsh, model, query_probes=[HV_A])

    def test_explicit_close_and_double_close(self, backend):
        lsh = backend.new_lsh()
        session = lsh.insertion_session()
        session.insert(b"a", minhash_of(HV_A))
        session.close()
        assert session.closed
        state_after_first_close = (
            set(lsh.keys.keys()),
            [set(t.keys()) for t in lsh.hashtables],
        )
        session.close()  # repeated close must not change anything
        session.close()
        assert (
            set(lsh.keys.keys()),
            [set(t.keys()) for t in lsh.hashtables],
        ) == state_after_first_close
        assert set(lsh.query(minhash_of(HV_A))) == {b"a"}

    def test_with_block_closes_once(self, backend):
        lsh = backend.new_lsh()
        with lsh.insertion_session() as session:
            session.insert(b"a", minhash_of(HV_A))
        assert session.closed
        session.close()  # no-op after the with block
        assert set(lsh.query(minhash_of(HV_A))) == {b"a"}

    def test_closed_session_rejects_operations(self, backend):
        lsh = backend.new_lsh()
        lsh.insert(b"a", minhash_of(HV_A))
        ins = lsh.insertion_session()
        ins.close()
        with pytest.raises(ValueError, match="closed"):
            ins.insert(b"b", minhash_of(HV_B))
        del_ = lsh.deletion_session()
        del_.close()
        with pytest.raises(ValueError, match="closed"):
            del_.remove(b"a")
        with pytest.raises(ValueError, match="closed"), ins:
            pass
        # The index itself is unaffected.
        assert set(lsh.query(minhash_of(HV_A))) == {b"a"}

    def test_close_and_with_are_equivalent(self, backend):
        results = []
        for close_style in ("with", "explicit"):
            lsh = backend.new_lsh()
            session = lsh.insertion_session(buffer_size=8)
            if close_style == "with":
                with session:
                    for i in range(10):
                        session.insert(f"k{i}".encode(), bulk_minhash(i))
            else:
                for i in range(10):
                    session.insert(f"k{i}".encode(), bulk_minhash(i))
                session.close()
            results.append(
                (
                    sorted(lsh.keys.keys()),
                    [sorted(t.keys()) for t in lsh.hashtables],
                    lsh.get_counts(),
                )
            )
        assert results[0] == results[1]

    def test_reread_with_new_object_same_basename(self, backend):
        """A new index object on the same basename reads the session result."""
        if backend.name == "dict":
            pytest.skip("dict storage does not share state between objects")
        basename = _basename()
        lsh = backend.new_lsh(basename=basename)
        with lsh.insertion_session(buffer_size=4) as session:
            session.insert(b"a", minhash_of(HV_A))
            session.insert(b"b", minhash_of(HV_B))
        with lsh.deletion_session(buffer_size=4) as session:
            session.remove(b"a")
        with lsh.insertion_session(buffer_size=4) as session:
            session.insert(b"c", minhash_of(HV_C))

        fresh = backend.new_lsh(basename=basename)
        model = IndexModel(fresh)
        model.add(b"b", HV_B)
        model.add(b"c", HV_C)
        assert_consistent(fresh, model, absent_keys=[b"a"], query_probes=[HV_A, HV_B, HV_C])


# ---------------------------------------------------------------------------
# Mixed workloads against the independent model
# ---------------------------------------------------------------------------


class TestMixedWorkload:
    @pytest.mark.parametrize("buffer_size", [2, 64, 4096])
    def test_sessions_and_direct_ops(self, backend, buffer_size):
        """A deterministic mixed script of session and direct maintenance,
        checking the full state against the independent model after every
        step.
        """
        lsh = backend.new_lsh()
        model = IndexModel(lsh)
        absent = set()

        def check(probes):
            assert_consistent(lsh, model, absent_keys=absent, query_probes=probes)

        # Phase 1: bulk insert 60 keys through a session.
        with lsh.insertion_session(buffer_size=buffer_size) as session:
            for i in range(60):
                session.insert(f"k{i}".encode(), bulk_minhash(i))
                model.add(f"k{i}".encode(), [i % 1000] * 4 + [i] * 4)
            # Duplicates of both flushed and pending keys are rejected.
            for i in (0, 30, 59):
                with pytest.raises(ValueError):
                    session.insert(f"k{i}".encode(), bulk_minhash(i + 5000))
        check([[0] * 4 + [0] * 4, [59 % 1000] * 4 + [59] * 4])

        # Phase 2: direct (non-session) removal and re-insertion.
        lsh.remove(b"k0")
        model.remove(b"k0")
        absent.add(b"k0")
        lsh.insert(b"k0", bulk_minhash(10_000))
        model.add(b"k0", [10_000 % 1000] * 4 + [10_000] * 4)
        absent.discard(b"k0")
        check([[10_000 % 1000] * 4 + [10_000] * 4])

        # Phase 3: delete every third key through a session, with errors in
        # between, and keep using the session after each error.
        with lsh.deletion_session(buffer_size=buffer_size) as session:
            for i in range(0, 60, 3):
                session.remove(f"k{i}".encode())
                model.remove(f"k{i}".encode())
                absent.add(f"k{i}".encode())
                with pytest.raises(ValueError):
                    session.remove(f"k{i}".encode())
                with pytest.raises(ValueError):
                    session.remove(b"never-there")
        absent.add(b"never-there")
        check([[1 % 1000] * 4 + [1] * 4, [2 % 1000] * 4 + [2] * 4])

        # Phase 4: re-insert half of the removed keys through a session.
        with lsh.insertion_session(buffer_size=buffer_size) as session:
            for i in range(0, 60, 6):
                session.insert(f"k{i}".encode(), bulk_minhash(i + 20_000))
                model.add(f"k{i}".encode(), [(i + 20_000) % 1000] * 4 + [i + 20_000] * 4)
                absent.discard(f"k{i}".encode())
        check([[20_000 % 1000] * 4 + [20_000] * 4, HV_FRESH])

        # Phase 5: delete everything; the index must be completely clean.
        with lsh.deletion_session(buffer_size=buffer_size) as session:
            for key in list(model.hashvalues):
                session.remove(key)
                model.remove(key)
        assert lsh.is_empty()
        assert lsh.get_counts() == [{}, {}]
        check([HV_A])


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_wrong_length_minhash(self, backend):
        lsh = backend.new_lsh()
        bad = MinHash(num_perm=NUM_PERM + 1)
        with pytest.raises(ValueError):
            lsh.insert(b"k", bad)
        with pytest.raises(ValueError):
            lsh.query(bad)
        with lsh.insertion_session() as session:
            with pytest.raises(ValueError):
                session.insert(b"k", bad)
            # The session is still usable.
            session.insert(b"k", minhash_of(HV_A))
        assert set(lsh.query(minhash_of(HV_A))) == {b"k"}

    def test_non_bytes_key_requires_prepickle(self, backend):
        lsh = backend.new_lsh()
        if backend.name == "dict" or backend.prepickle:
            # Non-bytes keys are fine here.
            lsh.insert("string-key", minhash_of(HV_A))
            assert "string-key" in lsh
        else:
            with pytest.raises(TypeError):
                lsh.insert("string-key", minhash_of(HV_A))
            with lsh.insertion_session() as session:
                with pytest.raises(TypeError):
                    session.insert("string-key", minhash_of(HV_A))
                session.insert(b"bytes-key", minhash_of(HV_A))
            assert b"bytes-key" in lsh


# ---------------------------------------------------------------------------
# Backend failure handling (fakeredis only: it can emulate disconnects)
# ---------------------------------------------------------------------------


class TestBackendFailure:
    def test_close_failure_propagates(self, backend):
        if not backend.name.startswith("fakeredis"):
            pytest.skip("fault injection uses fakeredis")
        lsh = backend.new_lsh()
        session = lsh.insertion_session(buffer_size=4096)
        session.insert(b"k1", minhash_of(HV_A))
        backend.server.connected = False
        with pytest.raises(redis_module.exceptions.ConnectionError):
            session.close()
        assert session.closed
        with pytest.raises(ValueError, match="closed"):
            session.insert(b"k2", minhash_of(HV_B))
        backend.server.connected = True
        # The failed flush discarded the buffered commands; nothing is
        # rolled back or replayed, and a new session keeps working.
        assert b"k1" not in lsh
        with lsh.insertion_session() as session2:
            session2.insert(b"k2", minhash_of(HV_B))
        assert set(lsh.query(minhash_of(HV_B))) == {b"k2"}

    def test_auto_flush_failure_propagates_and_session_recovers(self, backend):
        if not backend.name.startswith("fakeredis"):
            pytest.skip("fault injection uses fakeredis")
        lsh = backend.new_lsh()
        session = lsh.insertion_session(buffer_size=2)
        session.insert(b"k1", minhash_of(HV_A))
        session.insert(b"k2", minhash_of(HV_C))  # k1 is flushed here, k2 stays buffered
        backend.server.connected = False
        # Queueing the next key triggers an automatic flush, which fails.
        with pytest.raises(redis_module.exceptions.ConnectionError):
            session.insert(b"k3", minhash_of(HV_FRESH), check_duplication=False)
        backend.server.connected = True
        # The lost batch is not replayed: k2's key record is gone and can be
        # inserted again, while k1 made it to the storage and is a duplicate.
        with pytest.raises(ValueError, match="already exists"):
            session.insert(b"k1", minhash_of(HV_A))
        session.insert(b"k2", minhash_of(HV_C))
        session.insert(b"k3", minhash_of(HV_FRESH))
        session.close()
        assert set(lsh.query(minhash_of(HV_A))) == {b"k1"}
        assert set(lsh.query(minhash_of(HV_C))) == {b"k2"}
        assert set(lsh.query(minhash_of(HV_FRESH))) == {b"k3"}

    def test_deletion_close_failure_propagates(self, backend):
        if not backend.name.startswith("fakeredis"):
            pytest.skip("fault injection uses fakeredis")
        lsh = backend.new_lsh()
        lsh.insert(b"a", minhash_of(HV_A))
        session = lsh.deletion_session(buffer_size=4096)
        session.remove(b"a")
        backend.server.connected = False
        with pytest.raises(redis_module.exceptions.ConnectionError):
            session.close()
        assert session.closed
        backend.server.connected = True
        # The buffered removal was discarded with the failed flush; the key
        # is still present and can be removed again.
        assert b"a" in lsh
        lsh.remove(b"a")
        assert b"a" not in lsh


# ---------------------------------------------------------------------------
# Bookkeeping bounds
# ---------------------------------------------------------------------------


class TestBookkeeping:
    def test_pending_records_bounded_by_buffer(self, backend):
        """The in-memory bookkeeping must grow only with unflushed buffered
        work, not with the number of processed keys.
        """
        if backend.name == "dict":
            pytest.skip("dict storage keeps no pending records")
        buffer_size = 8
        lsh = backend.new_lsh()
        max_pending_inserts = 0
        max_stack = 0
        with lsh.insertion_session(buffer_size=buffer_size) as session:
            for i in range(500):
                session.insert(f"k{i}".encode(), bulk_minhash(i))
                max_pending_inserts = max(max_pending_inserts, len(lsh.keys._pending_inserts))
                max_stack = max(max_stack, len(lsh.keys._buffer.command_stack))
        assert max_pending_inserts <= buffer_size
        assert max_stack <= buffer_size
        # After the session, nothing is left over.
        assert lsh.keys._pending_inserts == set()
        assert lsh.keys._pending_removes == set()
        for table in lsh.hashtables:
            assert table._pending_srem_buckets == set()
            assert len(table._buffer.command_stack) == 0

        max_pending_removes = 0
        max_touched = 0
        with lsh.deletion_session(buffer_size=buffer_size) as session:
            for i in range(500):
                session.remove(f"k{i}".encode())
                max_pending_removes = max(max_pending_removes, len(lsh.keys._pending_removes))
                max_touched = max(max_touched, *(len(t._pending_srem_buckets) for t in lsh.hashtables))
        assert max_pending_removes <= buffer_size
        assert max_touched <= buffer_size
        assert lsh.is_empty()
        assert lsh.get_counts() == [{}, {}]


# ---------------------------------------------------------------------------
# Storage-level behaviour
# ---------------------------------------------------------------------------


class TestStorageLevel:
    def test_remove_val_drops_empty_entries(self, backend):
        """Storage-level remove_val must not leave zero-member entries in the
        key directory, for both the list and the set storage.
        """
        if backend.name == "dict":
            pytest.skip("redis storage specific")
        from datasketch.storage import RedisListStorage, RedisSetStorage

        config = backend.storage_config(_basename())
        list_storage = RedisListStorage(config, name=_basename())
        list_storage.insert(b"k", b"v1", b"v2")
        list_storage.remove_val(b"k", b"v1")
        assert list_storage.has_key(b"k")
        list_storage.remove_val(b"k", b"v2")
        assert not list_storage.has_key(b"k")
        assert list_storage.itemcounts() == {}

        set_storage = RedisSetStorage(config, name=_basename())
        set_storage.insert(b"H", b"m1", b"m2")
        set_storage.remove_val(b"H", b"m1")
        assert set_storage.get(b"H") == {b"m2"}
        set_storage.remove_val(b"H", b"m2")
        assert not set_storage.has_key(b"H")
        assert set_storage.itemcounts() == {}
