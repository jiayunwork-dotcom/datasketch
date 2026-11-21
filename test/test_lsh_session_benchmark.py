"""Observations of the batching cost of MinHashLSH maintenance sessions.

The suite drives insert -> delete -> re-insert phases over keys generated one
by one (the test never materializes the whole batch of MinHashes), with
duplicate requests scheduled around the automatic flush boundaries, and
records for every phase:

* how many write batches (pipeline executions) were performed,
* how many Redis commands were executed in total,
* how many network round trips that corresponds to (one per pipeline
  execution plus one per direct command),
* the peak extra Python memory (tracemalloc) and the largest pending-record
  set observed (the bookkeeping that tracks unflushed operations),
* the residual state after the session is closed.

The point is to show that the fix keeps writes batched (batches scale with
keys/buffer_size, not with keys), that acceptance decisions do not depend on
the buffer size, and that no bookkeeping outlives the session.

The workload size defaults to a few thousand keys so the test stays fast;
set DATASKETCH_BENCH_N=100000 for the full observation run. Real Redis runs
require a redis-server binary (see test_lsh_maintenance_sessions.py).
"""

import json
import os
import tracemalloc

import numpy as np
import pytest

from datasketch.lsh import MinHashLSH
from datasketch.minhash import MinHash
from test.test_lsh_maintenance_sessions import _basename, real_redis_server  # noqa: F401  (fixture re-export)

try:
    import redis as redis_module
except ImportError:  # pragma: no cover
    redis_module = None

try:
    import fakeredis
except ImportError:  # pragma: no cover
    fakeredis = None

NUM_PERM = 8
PARAMS = (2, 4)
N_KEYS = int(os.environ.get("DATASKETCH_BENCH_N", "3000"))
BUFFER_SIZES = (64, 4096)


def key_of(i):
    return f"bench-{i}".encode()


def minhash_of(i):
    # Band-0 buckets are shared in groups of 1000, band-1 buckets are
    # private; generated on the fly, never stored by the test.
    return MinHash(num_perm=NUM_PERM, hashvalues=np.array([i % 1000] * 4 + [i] * 4, dtype=np.uint64))


class CommandCounter:
    """Counts pipeline executions (batches), Redis commands and the network
    round trips they correspond to.
    """

    def __init__(self):
        self.batches = 0
        self.commands = 0
        self.round_trips = 0

    def install(self, monkeypatch, client_class):
        real_pipe_execute = redis_module.client.Pipeline.execute
        real_execute_command = client_class.execute_command

        def pipe_execute(pipe, *args, **kwargs):
            self.batches += 1
            self.commands += len(pipe.command_stack)
            self.round_trips += 1
            return real_pipe_execute(pipe, *args, **kwargs)

        def execute_command(client, *args, **kwargs):
            self.commands += 1
            self.round_trips += 1
            return real_execute_command(client, *args, **kwargs)

        monkeypatch.setattr(redis_module.client.Pipeline, "execute", pipe_execute)
        monkeypatch.setattr(client_class, "execute_command", execute_command)

    def snapshot(self):
        return (self.batches, self.commands, self.round_trips)

    def delta_since(self, snapshot):
        return (self.batches - snapshot[0], self.commands - snapshot[1], self.round_trips - snapshot[2])


def run_phase(*, lsh, mode, n, buffer_size, counter, events):
    """Run one insert/delete phase, scheduling duplicate requests so they
    land on both sides of automatic flushes, and collect the metrics.
    """
    storages = (lsh.keys, *lsh.hashtables)
    max_pending = 0
    tracemalloc.start()
    start = counter.snapshot()
    session_factory = lsh.insertion_session if mode == "insert" else lsh.deletion_session
    with session_factory(buffer_size=buffer_size) as session:
        for i in range(n):
            if mode == "insert":
                session.insert(key_of(i), minhash_of(i))
            else:
                session.remove(key_of(i))
            if i % 997 == 0:
                # Duplicate requests around the automatic flush boundaries:
                # the previous key is certainly accepted already, the key
                # buffer_size keys back may be flushed or still pending.
                try:
                    if mode == "insert":
                        session.insert(key_of(i), minhash_of(i))
                        events.append(("dup-current", i, "accepted"))
                    else:
                        session.remove(key_of(i))
                        events.append(("dup-current", i, "accepted"))
                except ValueError:
                    events.append(("dup-current", i, "rejected"))
                j = max(0, i - buffer_size)
                try:
                    if mode == "insert":
                        session.insert(key_of(j), minhash_of(j + 5_000_000))
                        events.append(("dup-earlier", i, "accepted"))
                    else:
                        session.remove(key_of(j))
                        events.append(("dup-earlier", i, "accepted"))
                except ValueError:
                    events.append(("dup-earlier", i, "rejected"))
            if i % 128 == 0:
                max_pending = max(
                    max_pending,
                    len(lsh.keys._pending_inserts),
                    len(lsh.keys._pending_removes),
                    *(len(t._pending_srem_buckets) for t in lsh.hashtables),
                )
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    batches, commands, round_trips = counter.delta_since(start)
    # A closed session leaves no residual bookkeeping or buffered commands.
    residual = {
        "pending_inserts": len(lsh.keys._pending_inserts),
        "pending_removes": len(lsh.keys._pending_removes),
        "pending_srem_buckets": sum(len(t._pending_srem_buckets) for t in lsh.hashtables),
        "buffered_commands": sum(len(s._buffer.command_stack) for s in storages),
    }
    return {
        "batches": batches,
        "commands": commands,
        "round_trips": round_trips,
        "peak_python_bytes": peak,
        "max_pending_records": max_pending,
        "residual": residual,
    }


@pytest.mark.parametrize("backend_name", ["fakeredis", "realredis"])
def test_session_batching_observation(backend_name, monkeypatch, request):
    if redis_module is None:
        pytest.skip("redis package not installed")
    server = None
    if backend_name == "fakeredis":
        if fakeredis is None:
            pytest.skip("fakeredis package not installed")
        server = fakeredis.FakeServer()
        monkeypatch.setattr(redis_module, "Redis", lambda **kw: fakeredis.FakeRedis(server=server, **kw))
        client_class = fakeredis.FakeRedis
        redis_params = {"host": "localhost", "port": 6399}
    else:
        server_params = request.getfixturevalue("real_redis_server")
        client_class = redis_module.Redis
        redis_params = dict(server_params)

    outcomes = {}
    for buffer_size in BUFFER_SIZES:
        counter = CommandCounter()
        # Each buffer size gets its own index namespace and its own counter.
        with monkeypatch.context() as m:
            counter.install(m, client_class)
            basename = _basename()
            lsh = MinHashLSH(
                num_perm=NUM_PERM,
                params=PARAMS,
                storage_config={
                    "type": "redis",
                    "basename": basename,
                    "redis": dict(redis_params),
                },
            )
            events = []
            metrics = {}
            for mode in ("insert", "delete", "re-insert"):
                phase_mode = "insert" if mode != "delete" else "delete"
                metrics[mode] = run_phase(
                    lsh=lsh,
                    mode=phase_mode,
                    n=N_KEYS,
                    buffer_size=buffer_size,
                    counter=counter,
                    events=events,
                )
            # Final state: all keys present exactly once, buckets consistent.
            assert lsh.keys.size() == N_KEYS
            counts = lsh.get_counts()
            assert len(counts[0]) == min(N_KEYS, 1000)  # shared band-0 buckets
            assert len(counts[1]) == N_KEYS  # private band-1 buckets
            assert all(size > 0 for band in counts for size in band.values())
            # Band-0 buckets are shared by keys with the same i % 1000.
            assert set(lsh.query(minhash_of(0))) == {key_of(i) for i in range(0, N_KEYS, 1000)}
            assert set(lsh.query(minhash_of(N_KEYS - 1))) == {
                key_of(i) for i in range(N_KEYS) if i % 1000 == (N_KEYS - 1) % 1000
            }
            if server is not None:
                probe = fakeredis.FakeRedis(server=server)
                # 1 keys-hash + N signature lists + 2 bucket hashes
                # + (1000 + N) bucket sets, all under this basename.
                namespace_keys = sum(1 for _ in probe.scan_iter(match=basename + b"*"))
                assert namespace_keys == 2 * N_KEYS + min(N_KEYS, 1000) + 3
            outcomes[buffer_size] = {"metrics": metrics, "events": events}

    # Acceptance decisions are identical for both buffer sizes.
    assert [e[2] for e in outcomes[64]["events"]] == [e[2] for e in outcomes[4096]["events"]]
    assert all(outcome == "rejected" for outcome in {e[2] for e in outcomes[64]["events"]})

    report = {"n_keys": N_KEYS, "backend": backend_name, "outcomes": outcomes}
    print("\n" + json.dumps(report, indent=2, sort_keys=True))  # noqa: T201 - the metrics report is the point

    for buffer_size, outcome in outcomes.items():
        metrics = outcome["metrics"]
        for phase in ("insert", "re-insert"):
            m = metrics[phase]
            # Writes stay batched: one flush per storage per buffer_size/2
            # keys, i.e. never anywhere near one batch per key.
            assert m["batches"] <= 3 * (2 * N_KEYS // buffer_size + 2), (buffer_size, phase, m)
            assert m["batches"] < N_KEYS // 8, (buffer_size, phase, m)
            # Bookkeeping is bounded by the buffer capacity.
            assert m["max_pending_records"] <= buffer_size, (buffer_size, phase, m)
        m = metrics["delete"]
        assert m["batches"] <= 3 * (2 * N_KEYS // buffer_size + 2) + 2 * (N_KEYS // buffer_size + 2), (
            buffer_size,
            m,
        )
        assert m["batches"] < N_KEYS // 8, (buffer_size, m)
        assert m["max_pending_records"] <= buffer_size, (buffer_size, m)
        for phase in ("insert", "delete", "re-insert"):
            assert metrics[phase]["residual"] == {
                "pending_inserts": 0,
                "pending_removes": 0,
                "pending_srem_buckets": 0,
                "buffered_commands": 0,
            }
