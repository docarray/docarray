# One Million Scale Benchmark

We created a DocumentArray with one million Documents and benchmark all supported document stores, including classic database and vector database:

* {ref}`In-memory<documentarray>`
* {ref}`Sqlite<sqlite>`
* {ref}`Qdrant<qdrant>`
* {ref}`Annlite<annlite>`
* {ref}`Weaviate<weaviate>`
* {ref}`ElasticSearch<elasticsearch>`

We focus on the following tasks:

1. **Create**: add new Documents to the document store via {meth}`~docarray.array.storage.base.seqlike.BaseSequenceLikeMixin.extend`.
2. **Read**: retrieve existing Documents from the document store by `.id`, i.e. `da['some_id']`.
3. **Update**: update existing Documents in the document store by `.id`, i.e. `da['some_id'] = Document(...)`.
4. **Delete**: delete Documents from the document store by `.id`, i.e. `del da['some_id']`
5. **Find by vector**: retrieve existing Documents via {meth}`~docarray.array.mixins.find` on `.embedding` using  nearest neighbor search or approximate nearest neighbor search, as described in {ref}`match-documentarray`.
6. **Find by condition**: search existing Documents via {meth}`~docarray.array.mixins.find` in the document store by boolean filters, as described in {ref}`find-documentarray`.

We are interested in the single-query performance on the above tasks, which means the tasks 2,3,4,5,6 are evaluated using one Document at a time, repeatedly, and report the average number.

The following table summarizes the result:

````{tab} Same HNSW parameters

| Store       | Create (s) | Read (ms) | Update (ms) | Delete (ms) | Find by condition (s) | Find by vector (s) | Recall@10 |
|---------------|-----------:|----------:|------------:|------------:|----------------------:|-------------------:|----------:|
| Memory        |  ** 0.6 ** | ** 0.1 ** |  ** 0.01 ** |  ** 0.39 ** |               ** 5 ** |               1.43 |      1.00 |
| Sqlite        |     4366.8 |       0.3 |        0.35 |        2.62 |                    31 |              23.59 |      1.00 |
| Annlite       |       72.4 |       0.3 |        6.57 |        4.62 |                    30 |               0.42 |      0.19 |
| Qdrant        |     2286.3 |       1.6 |        1.50 |        4.14 |                   605 |         ** 0.01 ** |      0.51 |
| Weaviate      |     1612.4 |      33.7 |       27.94 |       13.38 |                  1242 |               0.02 |      0.11 |
| ElasticSearch |     1307.3 |       2.4 |       40.85 |       40.74 |                   656 |               0.23 |  **0.85** |

````

````{tab} Default HNSW parameters

| Backend       | Create (s) | Read (ms) | Update (ms) | Delete (ms) | Find by condition (s) | Find by vector (s) | Recall@10 |
|---------------|-----------:|----------:|------------:|------------:|----------------------:|-------------------:|-------------------:|
| Memory        |  ** 0.6 ** | ** 0.1 ** |  ** 0.01 ** |  ** 0.16 ** |               ** 5 ** |               1.43 |               1.00 |
| Sqlite        |     4446.6 |       0.3 |        0.35 |       16.77 |                    30 |              24.38 |               1.00 |
| Annlite       |      114.0 |       0.3 |        9.36 |       20.09 |                    30 |               0.43 |               0.14 |
| Qdrant        |     2227.4 |       1.6 |       42.16 |       20.59 |                   608 |         ** 0.01 ** |               0.51 |
| Weaviate      |     1612.0 |       2.3 |       44.01 |       22.26 |                  1208 |         ** 0.01 ** |               0.10 |
| ElasticSearch |      715.2 |       2.1 |       15.58 |       33.26 |                   650 |               0.22 |           **0.83** |

````

## Benchmark setup

We now elaborate the setup of our benchmark. The benchmarking experiments used the following parameters:

| Parameter                                        | Value |
|--------------------------------------------------| --- |
| Number of created Documents                      | 1,000,000 |
| Number of Document on task 2,3,4,5,6             | 1 |
| The dimension of `.embedding`                    |  128 |
| Number of Documents for the task "Find by vector" | 10 |

We use the `Recall@K` value as an indicator of the search quality. The in-memory and SQLite store **do not implement** approximate nearest neighbor search but use exhaustive search instead. Hence, they give the maximum `Recall@K` but are the slowest. 


The experiments were conducted on a 4.5 Ghz AMD Ryzen Threadripper 3960X 24-Core Processor with Python 3.8.5 and using the official docker images of the storage backends. The docker images were allocated 40 GB of RAM.

### Set up on nearest neighbour search

As most of these document stores use their own implementation of HNSW (an approximate nearest neighbor search algorithm) but with different parameters, we conducted two sets of experiment for the "Find By Vector" task:
1. Set up HNSW with the same set of parameters; 
2. Set up HNSW with the default parameters specified by each vendor.

For the first experiment, the following HNSW parameters were fixed for all Document Stores that support ANN:
* **`ef=100` at construction**: ef at construction controls the index time/index accuracy. Bigger ef construction leads to longer construction, but better index quality.
* **`ef=100` at search**: the size of the dynamic list for the nearest neighbors. Higher ef at search leads to more accurate but slower search.
* **`m=16`**: max connections, the number of bi-directional links created for every new element during construction. Reasonable range for M is 2-100. Higher M work better on datasets with high intrinsic dimensionality and/or high recall, while low M work better for datasets with low intrinsic dimensionality and/or low recalls.

## Rationale

Our experiments are designed to be fair and the same across all backends while favouring Document Stores that benefit 
DocArray users the most. Note that such benchmark was impossible to set up without DocArray, as each store has its own API and the definition of a task varies from one to another. 


Our benchmark are designed based on the following principles:

* **Cover the most important operations**: We understand that some backends are better at some operations than others and 
some offer better quality. Therefore, we try to benchmark on 6 operations (CRUD + Find by vector + Find by condition)
and report quality measurement (Recall@K).
* **Not just the speed, quality also matters**: Since some backends offer better performance and some offer better quality, we make sure to report the quality measurement for the Approximate Nearest Neighbor Search. This will allow users to 
choose the backend that best suit their cases.
* **Same experiment, same API**: DocArray offers the same API across all backends and therefore, we built on top of it the 
same benchmarking experiment. Furthermore, we made sure to fix the same HNSW parameters for backends that support 
approximate nearest neighbor search. All backends are run on official Docker containers, local to the DocArray client 
which allows having similar network overhead. We also allocate the same resources for those Docker containers and all 
servers are run in a single node setup.
* **Benefit the user as much as possible**: We offer the same conditions and resources to all backends, but our experiment 
favours the backends that use the resources efficiently. Therefore, some backends might not use the network, or use 
GRPC instead of HTTP, or use batch operations. And we're okay with that, as long as it benefits the DocArray and Jina 
user.
* **Open to improvements**: We are constantly improving the performance of storage backends from the DocArray side and 
updating the benchmarks accordingly. If you believe we missed an optimization (e.g. perform an operation in batches, benefit 
from a recent feature in upstream, avoid unnecessary steps), feel free to raise a PR or issue, we're open to  your contributions!

## Incompleteness on stores

We do not yet cover the following backends for reasons:
* **Milvus**: currently DocArray does not integrate with Milvus. We're open for contributions to DocArray's repository to 
support it.
* **Pinecone**: Pinecone support is coming soon, and we'll add it to these benchmarks once available.
* **Faiss, Annoy, Scann**: We do not benchmark algorithms or ANN libraries. We only benchmark backends that can be used as 
Document stores. Actually we do not benchmark HNSW itself, but it is used by some backends internally.


## Conclusion

To conclude, one can look at those benchmarks and select the backend that suits better the use case.

Depending on the dataset size and the needed quality, some backends may be better than others. The following figure 
shows the variation of the time latency on the task "Find by vector" w.r.t. different backend according to the dataset size, with a time threshold of 0.3s, i.e. 300 ms:

```{figure} benchmark.svg
```

```{tip}
Sqlite backend is omitted from the plot as it is too slow to fit into this figure
```

If you're playing around on a dataset with less than 10k Documents, you can use default DocumentArray as-is (i.e. `memory` as storage) to enjoy the best 
quality for nearest neighbor search with reasonable time latency (less than 20 ms).

If your dataset does not fit in memory, and you **do not** care much about the speed of nearest neighbor search, you can use
`sqlite` as storage.

AnnLite is a good choice when indexing/appending/inserting speed matters more than the speed of finding by vectors. As AnnLite is a local monolith package that does not follow a client-server design, hence it avoids all network overhead.

Weaviate and Qdrant offer the fastest approximate nearest neighbor search, while ElasticSearch offers a good trade-off between speed and quality. ElasticSearch performs the best on the quality of ANN as we observed with highest Recall@K.
