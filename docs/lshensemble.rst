.. _minhash_lsh_ensemble:

MinHash LSH Ensemble
====================

.. _containment:

Containment
-----------

`Jaccard similarity <https://en.wikipedia.org/wiki/Jaccard_index>`_ is great for
measuring resemblance between two sets, however, it can be a biased measure for 
set intersection. This issue can be illustrated in the following Venn diagrams.
The left pair (Q and X) have roughly the same intersection size as the right 
pair (Q' and X').
But the Jaccard similarities, computed as 
:math:`\frac{|Q \cap X|}{|Q \cup X|}` and :math:`\frac{|Q' \cap X'|}{|Q' \cup X'|}`
respectively,
are very different, with the latter being much smaller, because its union
size :math:`|Q' \cup X'|` is much larger.
This shows that Jaccard similarity is biased when measuring intersection size, 
as large sets are penalized. 

.. figure:: /_static/containment.png
   :alt: MinHashLSHEnsemble Benchmark
   :scale: 80 %
   :align: center

We can use a better measure for intersection, called *containment*.
It is computed as the intersection size divided by the size of one of the set,
in this case Q.

.. math::
        Containment = \frac{|Q \cap X|}{|Q|}

It soon becomes clear why we use Q in the denominator, when you think about this 
search problem: suppose you have a large collection of sets, given a 
query, which is also a set, you want to find sets in your collection that have 
intersection with the query above a certain threshold.
We can think of the query as set Q, and an arbitrary set in the collection as X.
Since the query set here is fixed (for a specific search problem), the
denominator :math:`|Q|` is a constant, and the intersection size is solely determined
by the containment. Thus, we can search for sets in the collection with containment
above a certain threshold instead, and the containment threshold can be easily
deduced from the intersection threshold by multiplying :math:`|Q|`.

Another way to think about containment: it is a "normalized" intersection
(with value between 0 and 1), which measures the fraction of the query set Q
contained in X.

LSH for Containment
-------------------

Similar to :ref:`minhash_lsh`, there is an LSH index for containment search --
given a query set, find sets with containment above a certain threshold.
It is *LSH Ensemble* by `E. Zhu et al <http://www.vldb.org/pvldb/vol9/p1185-zhu.pdf>`_.
This package implements a slightly simplified version of the index,
:class:`datasketch.MinHashLSHEnsemble`.
The full implementation is in `Go <https://golang.org/>`_. It can be found at 
`github.com/ekzhu/lshensemble <https://github.com/ekzhu/lshensemble>`_.

Just like MinHash LSH, LSH Ensemble also works directly with :ref:`minhash` data sketches.

.. code:: python

        from datasketch import MinHashLSHEnsemble, MinHash

        set1 = set(["cat", "dog", "fish", "cow"])
        set2 = set(["cat", "dog", "fish", "cow", "pig", "elephant", "lion", "tiger",
                     "wolf", "bird", "human"])
        set3 = set(["cat", "dog", "car", "van", "train", "plane", "ship", "submarine",
                     "rocket", "bike", "scooter", "motorcyle", "SUV", "jet", "horse"])

        # Create MinHash objects
        m1 = MinHash(num_perm=128)
        m2 = MinHash(num_perm=128)
        m3 = MinHash(num_perm=128)
        for d in set1:
            m1.update(d.encode('utf8'))
        for d in set2:
            m2.update(d.encode('utf8'))
        for d in set3:
            m3.update(d.encode('utf8'))

        # Create an LSH Ensemble index with threshold and number of partition
        # settings.
        lshensemble = MinHashLSHEnsemble(threshold=0.8, num_perm=128, 
            num_part=32)

        # Index takes an iterable of (key, minhash, size)
        lshensemble.index([("m2", m2, len(set2)), ("m3", m3, len(set3))])

        # Check for membership using the key
        print("m2" in lshensemble)
        print("m3" in lshensemble)

        # Using m1 as the query, get an result iterator
        print("Sets with containment > 0.8:")
        for key in lshensemble.query(m1, len(set1)):
            print(key)

Paged queries
-------------

A hot bucket can hold thousands of keys, so iterating :meth:`query` to the
end can take a long time before a caller sees candidates from the other
partitions. For near-duplicate screening, where each candidate should be
checked as soon as it arrives, :meth:`MinHashLSHEnsemble.query_paged` returns
a cursor whose pages are measured in budget units instead::

        cursor = lshensemble.query_paged(m1, len(set1))
        while not cursor.done:
            page = cursor.next_page(64)   # caller-chosen positive budget
            for key in page.keys:
                recheck(key)              # inspect this material now
            print(page.budget_used, page.done)

A page charges one unit for every bucket lookup (an empty bucket still
costs one) and one for every key actually read from a bucket, including a
key that another band already delivered. The page stops as soon as the
budget is spent and keeps all read positions, so the next call resumes
instead of restarting. Each query delivers a given key at most once; read
to the end, the concatenated pages contain exactly the keys of
:meth:`query` on the same index. Pass an invalid budget (zero, negative,
non-integer) and the cursor raises without moving, so the corrected budget
continues from the same position.

Bucket lookups are scheduled before the bulk of bucket reads: band 0 of
every touched partition is probed first and a bucket found there yields a
key immediately, then the remaining bands are looked up in
partition-round-robin order, and only then are the buckets drained
round-robin. This keeps a small bucket in a later partition reachable
within a couple of pages even when another bucket has a thousand keys. The
cost is that a page can be empty (every key found was already delivered)
while the query is not yet done.

Paged queries need bucket storage that can be read lazily and resumed from
a position. The default in-memory ``dict`` storage supports this; other
backends are rejected with :class:`datasketch.storage.UnsupportedStorageError`
before any bucket is touched. The cursor lives in one process, so pages
cannot be resumed across processes. Use :meth:`MinHashLSHEnsembleQuery.close`
or a ``with`` block to stop early; after that no page can be read. Several
cursors over the same index can be advanced interleaved.

Merging shards
--------------

An LSH Ensemble partition layout is derived from the size distribution of the
indexed sets, so indexes built from different shards do not line up: the same
partition number can cover completely different size ranges. To combine them,
build each shard with ``mergeable=True`` and call :meth:`merge`:

.. code:: python

    left = MinHashLSHEnsemble(threshold=0.8, num_perm=128, num_part=32, mergeable=True)
    right = MinHashLSHEnsemble(threshold=0.8, num_perm=128, num_part=32, mergeable=True)
    left.index(entries_left)
    right.index(entries_right)

    combined = left.merge(right)              # a new, independent, mergeable index
    combined = combined.merge(third_shard)    # results can be merged again

``merge`` returns a *new* index. It recomputes the partition bounds from the
combined, de-duplicated size histogram and re-inserts every key into the
partition its size falls into, using the same flow as :meth:`index`.
Consequently the result's partition bounds, its per-query band selection, and
its candidate keys match an index built in a single shot from the same unique
entries; only key order is unspecified. The sources are not modified: the
result uses fresh bucket storage and a fresh key map, so it can be changed or
deleted without affecting the sources, and cursors already open over either
source keep reading.

Both indexes must have been built with ``mergeable=True``. An index built
without it keeps the historical, smaller memory profile and cannot be merged:
calling ``merge`` on it raises :class:`MergeNotSupportedError` before anything
is read or written. Only the default in-process ``dict`` storage is supported;
other backends raise :class:`datasketch.storage.UnsupportedStorageError` before
any content is touched.

The two indexes must share their construction configuration -- ``threshold``,
``num_perm``, ``num_part``, ``m``, the optimization ``weights``, and the key
encoding (``prepickle``) -- and their sketches must have been produced with the
same ``seed`` and permutation parameters, so their hash values live in the same
coordinate system. These checks are made even for empty shards. A shard that
mixed coordinate systems is rejected at :meth:`index` time. Only plain
:class:`MinHash` sketches are supported; ``WeightedMinHash`` and ``LeanMinHash``
are rejected.

Overlapping keys are controlled by ``overlap``:

.. code:: python

    combined = left.merge(right, overlap="error")       # fail on any shared key
    combined = left.merge(right, overlap="identical")   # default

``"error"`` raises as soon as any key is present in both indexes.
``"identical"`` accepts such a key once, only if its size and complete
``hashvalues`` are identical in both indexes; any difference raises
:class:`ValueError`. A shared key is never stored twice, and an accepted
overlap contributes to the size histogram only once. Key equality follows the
library's existing convention, so ``bytes``, ``str`` and ``tuple`` keys all
work under both the default encoding and ``prepickle=True``.

Memory and cost
~~~~~~~~~~~~~~~

Building a ``mergeable`` index retains each key's size and an immutable 8-byte
per-dimension snapshot of its hash values, i.e. ``8 * num_perm`` bytes per key
(the caller's mutable arrays are never kept). The merge itself re-hashes every
key once per band of its selected ``r``-level index -- ``O(num_keys *
num_perm / r)`` band hashes in total -- because the key's hash values, not its
old buckets, are the durable input: old buckets live in the old layout and
cannot be reused once the partition boundaries move. It does not build a
key-count by ``num_perm`` vector matrix: it streams the size histogram, then
inserts each key from its own snapshot. Peak additional memory is therefore
proportional to the number of keys times one snapshot plus the output index's
own buckets.


The precision of LSH Ensemble increases with the number of partitions, 
so does the querying time, as the plot below shows.
The experiment code can be found in the `benchmark` directory
of the source code repository.
There are other optional parameters that can be used to tune the index to achieve better accuracy or performance.
See the documentation of :class:`datasketch.MinHashLSHEnsemble` for details.

.. image:: /_static/lshensemble_benchmark_1k/lshensemble_num_perm_256_precision.png
   :scale: 75 %
   :align: center
   :alt: MinHashLSHEnsemble Benchmark Precision

.. image:: /_static/lshensemble_benchmark_1k/lshensemble_num_perm_256_recall.png
   :scale: 75 %
   :align: center
   :alt: MinHashLSHEnsemble Benchmark Recall

.. image:: /_static/lshensemble_benchmark_1k/lshensemble_num_perm_256_fscore.png
   :scale: 75 %
   :align: center
   :alt: MinHashLSHEnsemble Benchmark F-Score

.. image:: /_static/lshensemble_benchmark_1k/lshensemble_num_perm_256_query_time.png
   :scale: 75 %
   :align: center
   :alt: MinHashLSHEnsemble Benchmark Query Time


Common Issues with LSH Ensemble
-------------------------------

1. `Can I get containment values from LSH Ensemble? <https://github.com/ekzhu/datasketch/issues/97>`__
2. `How to use LSH Ensemble to find containment-based duplicates? <https://github.com/ekzhu/datasketch/issues/76#issuecomment-521275633>`__

