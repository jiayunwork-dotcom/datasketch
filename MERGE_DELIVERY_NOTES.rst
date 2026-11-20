Mergeable LSH Ensemble — delivery notes
=======================================

This document records how the "combine shards into one re-partitioned
MinHashLSHEnsemble" change was built, verified, and measured. It is scoped to
the work described here; unrelated pre-existing test failures are noted at the
end.

What changed
------------

1. ``MinHashLSHEnsemble(..., mergeable=True)``

   - A mergeable index keeps, per key, an immutable ``bytes`` snapshot of the
     sketch's hash values plus its size. The caller's mutable MinHash array is
     never retained.
   - The build path validates every sketch: it must be a plain ``MinHash`` with
     permutation parameters, the same ``seed`` and the same permutation arrays
     as the index. A shard that mixes coordinate systems is rejected at
     ``index`` time, before anything is inserted.
   - Indexes built without ``mergeable=True`` use the exact original code path
     and memory profile.

2. ``MinHashLSHEnsemble.merge(other, *, overlap="identical")``

   - Returns a **new** index. It re-derives the partition bounds from the
     combined, de-duplicated size histogram and re-inserts every key via the
     same flow as ``index``.
   - Checks construction config (``threshold``, ``num_perm``, ``num_part``,
     ``m``, ``weights``, ``prepickle``) and sketch coordinates (seed and
     permutations) on both indexes, including empty shards.
   - Rejects non-mergeable indexes with ``MergeNotSupportedError`` and non-dict
     storage with ``UnsupportedStorageError`` before touching content.
   - ``overlap="error"`` fails on any shared key; ``overlap="identical"``
     (default) accepts a shared key once only when size and full hash values
     match, else raises ``ValueError``. A shared key contributes to the size
     histogram once and is never stored twice.
   - The result is itself mergeable.

3. ``MinHashLSH.merge(other, check_overlap=False, *, overlap=...)``

   - ``overlap`` is a new keyword-only option. Omitting it keeps the historical
     ``check_overlap`` semantics exactly.
   - ``overlap="error"`` / ``"identical"`` cannot be combined with
     ``check_overlap=True``.
   - ``"identical"`` compares the full stored band-signature list for a key in
     both indexes (not a prefix), so a changed late band is caught.
   - All writes are planned before any is applied, so a late conflict leaves
     the receiver untouched; writes are rolled back on failure. Dict storage
     only; other backends are rejected up front.

Design choice: rebuild target buckets, keeping per-key snapshots
----------------------------------------------------------------

Two ways to move existing entries into a new partition layout were considered:

* **Reuse old band encodings.** A key's stored band signatures are per
  ``r``-level (band size) encodings. Within one index all partitions share the
  same set of ``r`` values, so a key's ``Hs`` for a given ``r`` can in
  principle be copied. But moving a key across partitions changes only which
  partition's containers hold it, not its ``r``; the buckets that must be
  written are new (different partition namespace). Crucially, the *ensemble*
  also changes each partition's band *count* ``b`` for a query, and a merged
  key may need a different ``r`` layout than its shard offered if it lands in a
  partition whose layout differs — the durable input is the hash values, not the
  old buckets. Hence this approach still has to know every key's hash values to
  fill the new buckets, so it only saves the band-hash recomputation, not the
  data movement or the partition decision.
* **Rebuild target buckets from per-key snapshots (chosen).** For each key we
  re-run the original insert with its retained hash values. Cost: one band-hash
  computation per band of every ``r``-level index in the key's target partition
  (``sum(num_perm // r)`` per key, measured 275 per key at ``num_perm=128``
  with the default parameter set), plus one partition assignment. It needs no
  key-count x ``num_perm`` matrix and no old-bucket reuse; peak extra memory is
  the size histogram plus the output index's own buckets, and the retained
  per-key snapshot is ``8 * num_perm`` bytes each.

Rebuilding is the only option that provably ties the result to a single-shot
build. The tests below compare bucket layouts, reverse key maps, band
selection, and candidate sets entry by entry against an index built in one
shot from the same unique entries, for several merge orders and both overlap
policies. That equivalence is what the requirement asks for, and it holds
regardless of how many keys move.

Verification (commands and results)
-----------------------------------

Environment: Python 3.12.16, numpy 2.5.5, scipy 1.18.1, pytest 9.2.0 (a
project venv at ``/tmp/venv312``).

Focus suites:

::

    python -m pytest test/test_lshensemble.py test/test_lsh.py \
        test/test_lshensemble_merge.py test/test_minhash.py \
        test/test_lean_minhash.py test/test_weighted_minhash.py \
        test/test_lshforest.py test/test_integration.py -q

Result (final):

::

    144 passed, 32 skipped in 9.73s

The 32 skips are the Redis/Cassandra integration variants (they require
``DO_TEST_REDIS`` / ``DO_TEST_CASSANDRA``); the dict-backed and mockredis
regressions all run.

The dedicated merge suite (``test/test_lshensemble_merge.py``) covers:

* two shards each with at least three non-empty partitions and disjoint size
  ranges (one small, one large), verified against a single-shot reference on
  partition bounds, band selection, buckets, reverse key map, and candidate
  sets, including queries at the old and new partition boundaries;
* three-shard merges in different orders and with swapped operands;
* empty shards on either side and both sides empty (queryable, empty);
* source MinHash arrays mutated after build;
* identical and conflicting overlaps, with a conflicting key arriving last;
* invalid configuration, mismatched seed, and mismatched permutation arrays,
  including for empty shards;
* default key encoding and ``prepickle=True``; ``str``, ``bytes`` and ``tuple``
  keys;
* rejection of ``WeightedMinHash`` / ``LeanMinHash`` and of mixed coordinate
  systems within a shard;
* result/source storage independence (no shared dicts; mutating one side does
  not affect the other);
* paged queries on the result and cursors opened on a source surviving the
  merge;
* an injected ``MemoryError`` during the result build leaves both sources
  unchanged and does not deliver a half-built result.

The LSH suite (``test/test_lsh.py::TestMergeOverlapPolicy``) covers:

* legacy semantics when ``overlap`` is omitted;
* ``overlap="error"`` on any intersection;
* ``overlap="identical"`` accepting equal signatures once and rejecting a
  changed first or last band;
* a conflict after non-conflicting keys leaving none of them written;
* ``check_overlap=True`` + ``overlap=`` rejection;
* non-dict storage rejected before modification;
* an injected ``MemoryError`` at a later band insert, verifying rollback of the
  receiver and that the source is unchanged;
* result equals a plain union when signatures match.

Benchmark (20,000 distinct keys, 128-dim sketches)
--------------------------------------------------

Script: ``benchmark/merge/lshensemble_merge_benchmark.py``. It generates
sketches one at a time (no pre-built vector matrix) and merges two
``mergeable`` shards of 10,000 keys each.

Measurement definitions:

* ``merge_seconds``: wall time of ``merge`` only, with no ``tracemalloc``
  attached.
* ``band_hashes_during_merge``: counted by wrapping ``MinHashLSH._byteswap``,
  i.e. actual band-hash computations during the merge.
* ``merge_peak_rss_bytes`` / ``merge_rss_delta_bytes``: peak resident set size
  sampled every 2 ms during the merge window (``/proc/self/statm``), and the
  delta over the pre-merge RSS.
* ``moved_ratio``: fraction of keys whose partition number differs from their
  shard-local partition.

Command:

::

    python benchmark/merge/lshensemble_merge_benchmark.py --keys 20000 --num-perm 128

Output (one clean run, each scenario in its own process):

::

    {"scenario": "stay", "keys": 20000, "num_perm": 128, "build_seconds": 131.865, "merge_seconds": 82.355, "band_hashes_during_merge": 5500000, "band_hashes_per_key": 275.0, "moved_keys": 2643, "moved_ratio": 0.1321, "rss_before_merge_bytes": 867475456, "merge_peak_rss_bytes": 1688776704, "merge_rss_delta_bytes": 821301248, "process_peak_rss_bytes": 1691774976, "merge_peak_python_bytes": null}
    {"scenario": "move", "keys": 20000, "num_perm": 128, "build_seconds": 238.747, "merge_seconds": 268.583, "band_hashes_during_merge": 5500000, "band_hashes_per_key": 275.0, "moved_keys": 18583, "moved_ratio": 0.9292, "rss_before_merge_bytes": 920698880, "merge_peak_rss_bytes": 1771630592, "merge_rss_delta_bytes": 850931712, "process_peak_rss_bytes": 1774628864, "merge_peak_python_bytes": null}

Interpretation:

* The band-hash count is the same in both scenarios (275 per key) because the
  merge hashes from hash values, not from old buckets: partition movement does
  not change the hashing work.
* The "stay" scenario (13% of keys move) merges in 82 s; the "move" scenario
  (93% move) takes 269 s. Profiling shows the dominant cost is
  ``optimal_partitions`` — the identical optimization a single-shot build
  performs on the combined data. Its cost scales with the number of *distinct*
  sizes squared (the NFP matrix): the "stay" data has ~4,900 distinct sizes and
  "move" has ~8,050 (~2.7x the matrix cells), which tracks the ~3.3x
  merge-time ratio. Bucket insertion is a small fraction. The merge is not
  accidentally quadratic in key count in my code; the quadratic term is the
  pre-existing partition optimizer that both a merge and a single-shot build
  already pay.
* Peak RSS delta (~820 MB "stay", ~850 MB "move") is dominated by the
  partition optimizer's NFP matrix (8 bytes per cell): ~4,900² cells ≈ 194 MB
  for "stay" and ~8,050² ≈ 519 MB for "move", plus the output index and the
  optimizer's working rows. The retained per-key snapshots are only
  ``8 * 128 = 1024`` bytes each (20,000 keys ≈ 20 MB). This is the same
  optimizer cost a single-shot ``index()`` would incur on the combined data.

To also record the Python-allocation peak, run with ``--trace-python-memory``.
``tracemalloc`` slows the merge several-fold (and increasingly so with key
count), so its ``merge_seconds`` is not comparable with the untraced numbers,
and it was therefore captured at a smaller size (2,000 keys) rather than
20,000:

::

    python benchmark/merge/lshensemble_merge_benchmark.py --keys 2000 --num-perm 128 --trace-python-memory

::

    {"scenario": "stay", "keys": 2000, "num_perm": 128, "build_seconds": 4.872, "merge_seconds": 40.396, "band_hashes_during_merge": 550000, "band_hashes_per_key": 275.0, "moved_keys": 794, "moved_ratio": 0.397, "rss_before_merge_bytes": 179863552, "merge_peak_rss_bytes": 328237056, "merge_rss_delta_bytes": 148373504, "process_peak_rss_bytes": 328658944, "merge_peak_python_bytes": 73927990}
    {"scenario": "move", "keys": 2000, "num_perm": 128, "build_seconds": 3.751, "merge_seconds": 21.12, "band_hashes_during_merge": 550000, "band_hashes_per_key": 275.0, "moved_keys": 1876, "moved_ratio": 0.938, "rss_before_merge_bytes": 177713152, "merge_peak_rss_bytes": 330514432, "merge_rss_delta_bytes": 152801280, "process_peak_rss_bytes": 331497472, "merge_peak_python_bytes": 78278966}

At 2,000 keys the Python-allocation peak is ~74-78 MB, dominated by the
partition optimizer's ``num_distinct_sizes x num_distinct_sizes`` NFP matrix
and its working rows, not by per-key snapshots (2,000 x 1,024 B ≈ 2 MB). This
matrix is the same one a single-shot ``index()`` builds; the merge does not add
a key-count x num_perm matrix on top of it.

Paths not exercised here
------------------------

* Redis and Cassandra storage backends are rejected by design for merging
  (``UnsupportedStorageError``), so no merge regression runs against them; the
  existing Redis query/paged regressions still pass under mockredis.
* The exploratory GPU (``cupy``) code path in ``MinHash.update_batch`` is
  unrelated and was not exercised.
* ``test/aio`` (motor/async LSH) is not installed in this environment and was
  not run; it is unrelated to this change.

Pre-existing failures (unrelated)
---------------------------------

``test/test_lshbloom.py`` fails to import ``pybloomfilter`` in this environment
(8 failures). Confirmed identical on the unmodified checkout.
