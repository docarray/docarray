# Benchmark

The script `scripts/benchmarking.py` benchmarks DocArray's supported Document Stores on 6 different operations:
* Create: Index Documents in the Document Store. This operation uses `DocumentArray.extend` which inserts new Documents in the storage.
* Read: Retrieve Documents from the Document Store by ID. 
* Update: Update Documents within the Document Store by ID.
* Delete: Delete Documents from the Document Store by ID.
* Find Document by vector: Nearest Neighbor Search or Approximate Nearest Neighbor Search by vector depending on the backend.
* Find Document by condition: Search Documents across the Document Store that satisfy a filter condition.

The following tables show the time cost to achieve those operations on 1 million Documents for indexing and 1 Document for query:

````{tab} Same HNSW parameters

| Backend       | Create (s) | Read (ms) | Update (ms) | Delete (ms) | Find by vector (s) | Recall at k=10 for | Find by condition (s) |
|---------------|-----------:|----------:|------------:|------------:|-------------------:|-------------------:|----------------------:|
| Memory        |  ** 0.6 ** | ** 0.1 ** |  ** 0.01 ** |  ** 0.39 ** |               1.43 |         ** 1.00 ** |               ** 5 ** |
| Sqlite        |     4366.8 |       0.3 |        0.35 |        2.62 |              23.59 |         ** 1.00 ** |                    31 |
| Annlite       |       72.4 |       0.3 |        6.57 |        4.62 |               0.42 |               0.19 |                    30 |
| Qdrant        |     2286.3 |       1.6 |        1.50 |        4.14 |         ** 0.01 ** |               0.51 |                   605 |
| Weaviate      |     1612.4 |      33.7 |       27.94 |       13.38 |               0.02 |               0.11 |                  1242 |
| ElasticSearch |     1307.3 |       2.4 |       40.85 |       40.74 |               0.23 |               0.85 |                   656 |

````

````{tab} Default HNSW parameters

| Backend       | Create (s) | Read (ms) | Update (ms) | Delete (ms) | Find by vector (s) | Recall at k=10 for | Find by condition (s) |
|---------------|-----------:|----------:|------------:|------------:|-------------------:|-------------------:|----------------------:|
| Memory        |  ** 0.6 ** | ** 0.1 ** |  ** 0.01 ** |  ** 0.16 ** |               1.43 |         ** 1.00 ** |               ** 5 ** |
| Sqlite        |     4446.6 |       0.3 |        0.35 |       16.77 |              24.38 |         ** 1.00 ** |                    30 |
| Annlite       |      114.0 |       0.3 |        9.36 |       20.09 |               0.43 |               0.14 |                    30 |
| Qdrant        |     2227.4 |       1.6 |       42.16 |       20.59 |         ** 0.01 ** |               0.51 |                   608 |
| Weaviate      |     1612.0 |       2.3 |       44.01 |       22.26 |         ** 0.01 ** |               0.10 |                  1208 |
| ElasticSearch |      715.2 |       2.1 |       15.58 |       33.26 |               0.22 |               0.83 |                   650 |

````

## Experimental setup

Since most of these Document Stores use their implementation of the HNSW Approximate Nearest Neighbor Search algorithm, 
with various default HNSW parameters, we conducted 2 benchmarking experiments for the `Find By Vector` operation:
1. Set up the Document Stores with the same HNSW parameters 
2. Set up the Document Stores with their default HNSW parameters at the time the benchmarking experiment was conducted.

Furthermore, we provide the `Recall@K` value, considering the exhaustive search as the ground truth. This allows taking 
into consideration the quality, not just the speed.

```{important}
The Sqlite and the in-memory Document Stores ** do not implement ** approximate nearest neighbor search and offer 
exhaustive search instead. That's why, they give the maximum quality but are the slowest when it comes to searching
 nearest neighbors.
```

The results were conducted on a 4.5 Ghz AMD Ryzen Threadripper 3960X 24-Core Processor with Python 3.8.5 and using the official docker 
images of the storage backends. The docker images were allocated 40 GB of RAM.

The benchmarking experiments used the following parameters:
* Number of indexed Documents: 1M
* Number of query Documents: 1
* Embedding dimension: 128
* Number of Documents (K) to be retrieved for `Find Document by vector`: 10

For the first experiment, the following HNSW parameters were fixed for the Document Stores that support ANN:
* `ef=100` at construction: ef at construction controls the index time/index accuracy. Bigger ef construction leads to longer construction, but better index quality.
* `ef=100` at search: the size of the dynamic list for the nearest neighbors. Higher ef at search leads to more accurate but slower search.
* `m=16`: max connections, the number of bi-directional links created for every new element during construction. Reasonable range for M is 2-100. Higher M work better on datasets with high intrinsic dimensionality and/or high recall, while low M work better for datasets with low intrinsic dimensionality and/or low recalls.

## Rationale
Our experiments are designed to be fair and the same across all backends while favouring Document Stores that benefit 
the DocArray user. Therefore, the benchmarks are based on the following:

* Cover the most important operations: We understand that some backends are better at some operations than others and 
some offer better quality. Therefore, we try to benchmark on 6 operations (CRUD + Find by vector + Find by condition)
and report quality measurement (Recall at K).
* Not just the speed, quality in mind: Since some backends offer better performance and some offer better quality, 
we make sure to report the quality measurement for the Approximate Nearest Neighbor Search. This will allow users to 
choose the backend that best suit their cases.
* Same experiment, same API: DocArray offers the same API across all backends and therefore, we built on top of it the 
same benchmarking experiment. Furthermore, we made sure to fix the same HNSW parameters for backends that support 
Approximate Nearest Neighbor Search. All backends are run on official Docker containers, local to the DocArray client 
which allows having similar network overhead. We also allocate the same resources for those Docker containers and all 
servers are run in a single node setup.
* Benefit the user as much as possible: We offer the same conditions and resources to all backends, but our experiment 
favours the backends that use the resources efficiently. Therefore, some backends might not use the network, or use 
GRPC instead of HTTP, or use batch operations. And we're okay with that, as long as it benefits the DocArray and Jina 
user.
* Open to improvements: We are constantly improving the performance of storage backends from the DocArray side and 
updating the benchmarks accordingly. If you believe we missed an optimization (perform an operation in batches, benefit 
from a recent feature in upstream, avoid unnecessary steps,...), feel free to raise a PR or issue, we're open to 
your contributions!

## Backends covered
Our benchmarks cover all backends supported by DocArray:
* {ref}`In-memory<documentarray>`
* {ref}`Sqlite<sqlite>`
* {ref}`Qdrant<qdrant>`
* {ref}`Annlite<annlite>`
* {ref}`Weaviate<weaviate>`
* {ref}`ElasticSearch<elasticsearch>`

We do not cover the following backends for those reasons:
* Milvus: currently DocArray does not integrate with Milvus. We're open for contributions to DocArray's repository to 
support it.
* Pinecone: Pinecode support is coming soon, and we'll add it to these benchmarks once available.
* Faiss, Annoy, Scann: We do not benchmark algorithms or ANN libraries. We only benchmark backends that can be used as 
Document stores. Actually we do not benchmark HNSW itself, but it is used by some backends internally.


## Conclusion
To conclude, one can look at those benchmarks and select the backend that suits better the use case.

Depending on the dataset size and the needed quality, some backends may be better than others. The following figure 
shows the variation of the `Find by vector` time latency per backend according to the dataset size, with a time 
threshold of 300 ms:

```{figure} benchmark.svg
```
**Note**: Sqlite backend was omitted from the plot.

If you're playing around on a dataset with less than 10k Documents, you can use `memory` as storage to enjoy the best 
quality for Nearest Neighbor Search with reasonable time latency (less than 20 ms).

If your dataset does not fit in memory and you don't care much about the speed of Nearest Neighbor Search, you can use
`sqlite` as storage.

AnnLite does not implement a client-server pattern and therefore, it does not present network overhead and allows fast 
indexing. It also supports Approximate Nearest Neighbor Search.

Weaviate offers fast Approximate Nearest Neighbor Search while ElasticSearch and Qdrant offer good compromise between 
speed and quality. On the other hand, ElasticSearch performs better in terms of quality.
