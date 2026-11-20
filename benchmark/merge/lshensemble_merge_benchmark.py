"""Benchmark merging MinHashLSHEnsemble shards into a re-partitioned index.

What is measured
----------------
For each scenario two ``mergeable=True`` indexes are built from two shards, then
combined with ``MinHashLSHEnsemble.merge``. Reported per scenario:

* ``build_seconds``: wall time to build both shards (not included in the merge
  time, but reported so the merge cost can be compared with the cost of
  building the same data from scratch).
* ``merge_seconds``: wall time of ``merge`` alone, measured with
  ``time.perf_counter`` and *without* ``tracemalloc`` attached (tracing every
  allocation slows the merge down by an order of magnitude and would distort
  the number).
* ``band_hashes_during_merge``: number of band hash computations the merge
  actually performs, counted by wrapping ``MinHashLSH._byteswap``. Each indexed
  key contributes one call per band of every ``r``-level index in its target
  partition, i.e. ``sum(num_perm // r for r in unique_rs)`` per key -- this is
  the real hashing work, not an estimate.
* ``moved_keys`` / ``moved_ratio``: keys whose partition number differs from
  the partition they had in their source shard.
* ``merge_peak_rss_bytes``: native peak resident set size observed while the
  merge runs. A sampler thread reads ``/proc/self/statm`` every 2 ms during the
  merge window and keeps the maximum; this isolates the merge's window from
  the shard build that precedes it. ``process_peak_rss_bytes`` is
  ``ru_maxrss`` for the whole process (monotonic, includes shard build), and
  ``merge_peak_python_bytes`` is ``tracemalloc``'s peak from an optional
  ``--trace-python-memory`` run, so Python-level allocation can be seen
  separately. Enabling that flag slows the merge, so ``merge_seconds`` from a
  traced run is not comparable with an untraced one; run it twice if both a
  clean time and a Python-allocation peak are wanted.

Input construction
------------------
Sketches are generated one at a time: a generator yields each
``(key, MinHash, size)`` from tokens built on the fly, and the fixture never
holds a key-count x num_perm array (no ``MinHash.bulk``). The index takes its
own immutable snapshot per key; the fixture itself only ever holds one sketch.

Scenarios
---------
``stay``:  both shards draw from one size distribution, so merged boundaries
           stay close to each shard's and few keys move.
``move``:  shard A holds small sets and shard B large sets, so the combined
           histogram re-cuts every boundary and most keys move.

Each scenario runs in its own process (``--scenario stay|move``) so RSS peaks
are not shared. Run all scenarios with::

    python benchmark/merge/lshensemble_merge_benchmark.py --keys 20000 --num-perm 128

which spawns one fresh interpreter per scenario and prints one JSON line each.
"""

import argparse
import json
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from datasketch import MinHash, MinHashLSHEnsemble


def _emit(record):
    """Print one machine-readable result record (the benchmark's output format)."""
    print(json.dumps(record))  # noqa: T201 - this is the script's output


def _sketch_for(key, size, num_perm):
    """Build one num_perm-dim MinHash from a short token list, no bulk arrays."""
    minhash = MinHash(num_perm=num_perm, seed=1)
    # A shared pool plus a few key-specific tokens, so queries are meaningful.
    tokens = [b"pool-%d" % (i % 97) for i in range(min(size, 97))]
    tokens += [("%s-%d" % (key, i)).encode() for i in range(min(size, 3))]
    minhash.update_batch(tokens)
    return minhash


def _entries(prefix, sizes, num_perm, batch=128):
    """Yield entries one batch at a time; the sketch list is never materialized."""
    for start in range(0, len(sizes), batch):
        for offset in range(start, min(start + batch, len(sizes))):
            key = "%s-%d" % (prefix, offset)
            yield key, _sketch_for(key, sizes[offset], num_perm), sizes[offset]


def _partitions(ensemble):
    mapping = {}
    for partition, index in enumerate(ensemble.indexes):
        keys = set()
        for lsh in index.values():
            keys.update(lsh.keys)
        for key in keys:
            mapping.setdefault(key, set()).add(partition)
    return mapping


def _count_merge_hashes():
    """Return ``(context manager, counter)`` counting band hashes during merge.

    The merge builds a new index internally, so wrapping the sources' ``_H``
    would count nothing. A new ``MinHashLSH`` binds ``self._byteswap`` at
    construction, so patching the class method before the merge routes every
    band hash of the new index through the counter.
    """
    from unittest.mock import patch as mock_patch

    from datasketch.lsh import MinHashLSH

    counter = {"n": 0}
    original = MinHashLSH._byteswap

    def counting(self, hs):
        counter["n"] += 1
        return original(self, hs)

    return mock_patch.object(MinHashLSH, "_byteswap", counting), counter


_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def _rss_bytes():
    """Current resident set size from /proc/self/statm, or None if unavailable."""
    try:
        with open("/proc/self/statm") as handle:
            resident_pages = int(handle.read().split()[1])
    except (OSError, IndexError, ValueError):
        return None
    return resident_pages * _PAGE_SIZE


class _RssPeakSampler:
    """Sample RSS in a background thread and keep the maximum over a window."""

    def __init__(self, interval=0.002):
        self._interval = interval
        self._stop = threading.Event()
        self.peak = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            value = _rss_bytes()
            if value is not None and (self.peak is None or value > self.peak):
                self.peak = value
            self._stop.wait(self._interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop.set()
        self._thread.join()
        return False


def _process_peak_rss_bytes():
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS bytes.
    return peak if sys.platform == "darwin" else peak * 1024


def _build(entries, num_perm):
    ensemble = MinHashLSHEnsemble(threshold=0.5, num_perm=num_perm, num_part=16, m=8, mergeable=True)
    ensemble.index(entries)
    return ensemble


def run_scenario(name, sizes_a, sizes_b, num_perm, trace_python_memory=False):
    started = time.perf_counter()
    left = _build(_entries("a", sizes_a, num_perm), num_perm)
    right = _build(_entries("b", sizes_b, num_perm), num_perm)
    build_seconds = time.perf_counter() - started

    before = _partitions(left)
    before.update(_partitions(right))

    hash_counter, counter = _count_merge_hashes()
    python_peak = None
    if trace_python_memory:
        import tracemalloc

        tracemalloc.start()
        tracemalloc.reset_peak()

    rss_before = _rss_bytes()
    with _RssPeakSampler() as sampler, hash_counter:
        started = time.perf_counter()
        merged = left.merge(right)
        merge_seconds = time.perf_counter() - started

    if trace_python_memory:
        _, python_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    after = _partitions(merged)
    moved = sum(1 for key, parts in after.items() if before.get(key) != parts)
    total = len(after)
    return {
        "scenario": name,
        "keys": total,
        "num_perm": num_perm,
        "build_seconds": round(build_seconds, 3),
        "merge_seconds": round(merge_seconds, 3),
        "band_hashes_during_merge": counter["n"],
        "band_hashes_per_key": round(counter["n"] / total, 2) if total else 0,
        "moved_keys": moved,
        "moved_ratio": round(moved / total, 4) if total else 0.0,
        "rss_before_merge_bytes": rss_before,
        "merge_peak_rss_bytes": sampler.peak,
        "merge_rss_delta_bytes": (sampler.peak - rss_before) if sampler.peak and rss_before else None,
        "process_peak_rss_bytes": _process_peak_rss_bytes(),
        "merge_peak_python_bytes": python_peak,
    }


def scenario_sizes(name, keys, seed):
    """Return ``(sizes_a, sizes_b)`` for a scenario."""
    half = keys // 2
    rng = np.random.default_rng(seed)
    if name == "stay":
        shared = rng.integers(1, 5000, keys)
        return shared[:half].tolist(), shared[half:].tolist()
    return rng.integers(10, 200, half).tolist(), rng.integers(20000, 40000, half).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys", type=int, default=20000, help="total distinct keys (split across two shards)")
    parser.add_argument("--num-perm", type=int, default=128, help="MinHash dimensions")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--scenario",
        choices=["stay", "move", "all"],
        default="all",
        help="which scenario to run; each scenario runs in its own process so RSS peaks are clean",
    )
    parser.add_argument(
        "--trace-python-memory",
        action="store_true",
        help="also measure a tracemalloc Python-allocation peak (slower; timing is taken from the untraced run)",
    )
    args = parser.parse_args()

    def run_one(name):
        sizes_a, sizes_b = scenario_sizes(name, args.keys, args.seed)
        _emit(run_scenario(name, sizes_a, sizes_b, args.num_perm, args.trace_python_memory))

    if args.scenario != "all":
        run_one(args.scenario)
        return

    # One fresh interpreter per scenario so native RSS peaks are not shared.
    import subprocess

    for name in ("stay", "move"):
        command = [
            sys.executable,
            os.path.abspath(__file__),
            "--keys",
            str(args.keys),
            "--num-perm",
            str(args.num_perm),
            "--seed",
            str(args.seed),
            "--scenario",
            name,
        ]
        if args.trace_python_memory:
            command.append("--trace-python-memory")
        subprocess.run(command, check=True)  # noqa: S603 - arguments are built here


if __name__ == "__main__":
    main()
