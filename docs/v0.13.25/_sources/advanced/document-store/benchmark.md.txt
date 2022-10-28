# One Million Scale Benchmark


```{figure} https://docarray.jina.ai/_images/benchmark-banner.gif
:width: 0 %
:scale: 0 %
```

```{figure} benchmark-banner.gif
:scale: 0 %
```

We create a DocumentArray with one million Documents using randomly generated data from a uniform distribution and benchmark all supported document stores.
This includes classic database and vector database, all under the same DocumentArray API:

| Name                                                                                        | Usage                                    | Version           |
|---------------------------------------------------------------------------------------------|------------------------------------------|-------------------|
| [`"None"`](../../../fundamentals/documentarray/#documentarray), namely an in-memory "store" | `DocumentArray()`                        | DocArray `0.12.8` |
| [`Sqlite`](../sqlite/#sqlite)                                                               | `DocumentArray(storage='sqlite')`        | `2.6.0`           |
| [`Weaviate`](../weaviate/#weaviate)                                                         | `DocumentArray(storage='weaviate')`      | `1.11.0`          |
| [`Qdrant`](../qdrant/#qdrant)                                                               | `DocumentArray(storage='qdrant')`        | `0.7.0`           |
| [`Annlite`](../annlite/#annlite)                                                            | `DocumentArray(storage='anlite')`        | `0.3.1`           |
| [`ElasticSearch`](../elasticsearch/#elasticsearch)                                          | `DocumentArray(storage='elasticsearch')` | `8.1.0`           |


We focus on the following tasks:

1. **Create**: add one million Documents to the document store via {meth}`~docarray.array.storage.base.seqlike.BaseSequenceLikeMixin.extend`.
2. **Read**: retrieve existing Documents from the document store by `.id`, i.e. `da['some_id']`.
3. **Update**: update existing Documents in the document store by `.id`, i.e. `da['some_id'] = Document(...)`.
4. **Delete**: delete Documents from the document store by `.id`, i.e. `del da['some_id']`
5. **Find by condition**: search existing Documents by `.tags` via {meth}`~docarray.array.mixins.find` in the document store by boolean filters, as described in {ref}`find-documentarray`.
6. **Find by vector**: retrieve existing Documents by `.embedding` via {meth}`~docarray.array.mixins.find`  using  nearest neighbor search or approximate nearest neighbor search, as described in {ref}`match-documentarray`.

The above tasks are often atomic operations in high-level DocArray API. Hence, understanding their performance gives user a good estimation of the experience when using DocumentArray with different backends.

We are interested in the single query performance on the above tasks, which means tasks 2,3,4,5,6 are evaluated using one Document at a time, repeatedly. We report the average number.


```{attention}

* **Benchmarks are conducted end-to-end**: We benchmark function calls from DocArray, not just the underlying backend vector database. Therefore, results for a particular backend can be influenced (positively or negatively) by our interface. If you can spot bottlenecks we would be thrilled to know about and improve our code.
* **We use similar underlying search algorithms but different implementations**: In this benchmark we focus on setting only parameters `ef`, `ef_construct` and `max_connections` from HNSW. Note that there might be other parameters that storage backends can fix than might or might not be accessible and can have a big impact on performance. This means that even similar configurations cannot be easily compared.
* **Benchmark is for DocArray users, not for research**: This benchmark showcases what a user can expect to get from DocArray without tuning hyper-parameters of a vector database. In practice, we strongly recommend tuning them to achieve high quality results.
```

## Benchmark result

The following table summarizes the result. The smaller the values, the better (except for `Recall@10`). The best performer of each task is highlighted in **bold**:

````{tab} Same HNSW parameters (ef, ef_construct, max_connections)

| Store         | Create 1M (s) | Read (ms) | Update (ms) | Delete (ms) | Find by condition (s) | Find by vector (s) | Recall@10 |
|---------------|-----------:|----------:|------------:|------------:|----------------------:|-------------------:|----------:|
| None        |  ** 0.6 ** | ** 0.1 ** |  ** 0.01 ** |  ** 0.39 ** |               ** 5 ** |               1.43 |      1.00 |
| Sqlite        |    4,366.8 |       0.3 |        0.35 |        2.62 |                    31 |              23.59 |      1.00 |
| Annlite       |       72.4 |       0.3 |        6.57 |        4.62 |                    30 |               0.42 |      0.19 |
| Qdrant        |    2,286.3 |       1.6 |        1.50 |        4.14 |                   605 |         ** 0.01 ** |      0.51 |
| Weaviate      |    1,612.4 |      33.7 |       27.94 |       13.38 |                 1,242 |               0.02 |      0.11 |
| ElasticSearch |    1,307.3 |       2.4 |       40.85 |       40.74 |                   656 |               0.23 |  **0.85** |

````



````{tab} Default HNSW parameters

| Store       | Create 1M (s) | Read (ms) | Update (ms) | Delete (ms) | Find by condition (s) | Find by vector (s) | Recall@10 |
|---------------|-----------:|----------:|------------:|------------:|----------------------:|-------------------:|----------:|
| None        |  ** 0.6 ** | ** 0.1 ** |  ** 0.01 ** |  ** 0.16 ** |               ** 5 ** |               1.43 |      1.00 |
| Sqlite        |    4,446.6 |       0.3 |        0.35 |       16.77 |                    30 |              24.38 |      1.00 |
| Annlite       |      114.0 |       0.3 |        9.36 |       20.09 |                    30 |               0.43 |      0.14 |
| Qdrant        |    2,227.4 |       1.6 |       42.16 |       20.59 |                   608 |         ** 0.01 ** |      0.51 |
| Weaviate      |    1,612.0 |       2.3 |       44.01 |       22.26 |                 1,208 |         ** 0.01 ** |      0.10 |
| ElasticSearch |      715.2 |       2.1 |       15.58 |       33.26 |                   650 |               0.22 |  **0.83** |

````

When we consider each query as a Document, we can convert the above metrics into query/document per second, i.e. QPS/DPS. Values are higher the better (except for `Recall@10`). The best performer of each task is highlighted in **bold**:


````{tab} Same HNSW parameters (ef, ef_construct, max_connections) in QPS

| Store         |          Create 1M |        Read |       Update |      Delete | Find by condition | Find by vector | Recall@10 |
|---------------|----------------:|------------:|-------------:|------------:|------------------:|---------------:|----------:|
| None        | ** 1,610,305 ** | ** 9,345 ** | ** 71,428 ** | ** 2,570 ** |       ** 0.190 ** |           0.70 |      1.00 |
| Sqlite        |              22 |        3125 |        2,816 |         381 |             0.032 |           0.04 |      1.00 |
| Annlite       |           1,380 |       4,000 |          152 |         216 |             0.033 |           2.34 |      0.19 |
| Qdrant        |              43 |         604 |          664 |         241 |             0.002 |    ** 90.90 ** |      0.51 |
| Weaviate      |              62 |          31 |           35 |          74 |             0.001 |          55.55 |      0.11 |
| ElasticSearch |              76 |         409 |           24 |          24 |             0.002 |           4.39 |  **0.85** |

````


````{tab} Default HNSW parameters in QPS

| Store         |          Create 1M |         Read |       Update |      Delete | Find by condition | Find by vector | Recall@10 |
|---------------|----------------:|-------------:|-------------:|------------:|------------------:|---------------:|----------:|
| None        | ** 1,618,122 ** | ** 10,101 ** | ** 71,428 ** | ** 6,211 ** |       ** 0.192 ** |           0.70 |      1.00 |
| Sqlite        |             224 |        3,125 |        2,824 |          59 |             0.033 |           0.04 |      1.00 |
| Annlite       |           8,769 |        3,759 |          106 |          49 |             0.033 |           2.34 |      0.14 |
| Qdrant        |             448 |          615 |           23 |          48 |             0.002 |    ** 111.1 ** |      0.51 |
| Weaviate      |             620 |          435 |           22 |          44 |             0.001 |          200.0 |      0.10 |
| ElasticSearch |           1,398 |          481 |           64 |          30 |             0.002 |            4.5 |  **0.83** |

````

## Benchmark setup

We now elaborate the setup of our benchmark. First the following parameters are used:

| Parameter                                       | Value     |
|-------------------------------------------------|-----------|
| Number of created Documents                     | 1,000,000 |
| Number of Document on task 2,3,4,5,6            | 1         |
| The dimension of `.embedding`                   | 128       |
| Number of results for the task "Find by vector" | 10        |

We choose 1 million Documents for two reasons: (1) it is the most common data scale for SME. (2) we expect it is a performance-wise breaking point for a production system.

Each Document follows the structure: 

```json
{
  "id": "94ee6627ee7f582e5e28124e78c3d2f9",
  "tags": {"i": 10},
  "embedding": [0.49841760378680844, 0.703959752118305, 0.6920759535687985, 0.10248648858410625, ...]
}
```


We use `Recall@K` value as an indicator of the search quality. The in-memory and SQLite store **do not implement** approximate nearest neighbor search but use exhaustive search instead. Hence, they give the maximum `Recall@K` but are the slowest. 

The experiments were conducted on a 4.5 Ghz AMD Ryzen Threadripper 3960X 24-Core Processor with Python 3.8.5 and DocArray 0.12.8.

Besides, as Weaviate, Qdrant and ElasticSearch follow a client/server pattern, we set up them with their official 
docker images in a **single node** configuration, with 40 GB of RAM allocated. That is, only 1 replica and shard are 
operated during the benchmarking. We did not opt for a cluster setup because our benchmarks mainly aim to assess the 
capabilities of a single instance of the server.

Results might include overhead coming from DocArray side which applies equally for all backends, unless a specific 
backend provides a more efficient implementation.

### Settings of the nearest neighbour search

As most of these document stores use their own implementation of HNSW (an approximate nearest neighbor search algorithm) but with different parameters, we conducted two sets of experiment for the "Find By Vector" task:
1. Set up HNSW with the same set of parameters; 
2. Set up HNSW with the default parameters specified by each vendor;

For the first experiment, the following HNSW parameters were fixed for all Document Stores that support ANN:
* **`ef=100` at construction**: `ef` at construction controls the index time/index accuracy. Bigger `ef` construction leads to longer construction, but better index quality.
* **`ef=100` at search**: the size of the dynamic list for the nearest neighbors. Higher `ef` at search leads to more accurate but slower search.
* **`m=16`**: max connections, the number of bi-directional links created for every new element during construction. Reasonable range for `m` is 2-100. Higher `m` work better on datasets with high intrinsic dimensionality and/or high recall, while low `m` work better for datasets with low intrinsic dimensionality and/or low recall.


Finally, the full benchmark script is [available here](https://github.com/jina-ai/docarray/blob/main/scripts/benchmarking.py).

### Rationale on the experiment design

Our experiments are designed to be fair and the same across all backends while favouring document stores that benefit 
DocArray users the most. Note that such a benchmark was impossible to set up before DocArray, as each store has its own API and the definition of a task varies. 

Our benchmark is based on the following principles:

* **Cover the most important operations**: We understand that some backends are better at some operations than others, and 
some offer better quality. Therefore, we try to benchmark on 6 operations (CRUD + Find by vector + Find by condition)
and report quality measurement (`Recall@K`).
* **Not just speed, but also quality**: Since some backends offer better performance and some offer better quality, we make sure to report the quality measurement for the approximate nearest neighbor search. This will allow users to 
choose the backend that best suits their cases.
* **Same experiment, same API**: DocArray offers the same API across all backends and therefore we built on top of it the 
same benchmarking experiment. Furthermore, we made sure to fix the same HNSW parameters for backends that support 
approximate nearest neighbor search. All backends are run on official Docker containers, local to the DocArray client 
which allows having similar network overhead. We also allocate the same resources for those Docker containers and all 
servers are run in a single node setup.
* **Benefit the user as much as possible**: We offer the same conditions and resources to all backends, but our experiment 
favours the backends that use the resources efficiently. Therefore, some backends might not use the network, or use 
GRPC instead of HTTP, or use batch operations. We're okay with that, as long as it benefits the DocArray and Jina 
user.
* **Open to improvements**: We are constantly improving the performance of storage backends from the DocArray side and 
updating the benchmarks accordingly. If you believe we missed an optimization (e.g. perform an operation in batches, benefit from a recent feature in upstream, avoid unnecessary steps), feel free to [raise a PR or issue](https://github.com/jina-ai/docarray/issues/new/choose). We're open to your contributions!

### Incompleteness on the stores

We do not yet cover the following backends for various reasons:
* **Milvus**: currently DocArray does not integrate with Milvus. We're open for contributions to DocArray's repository to 
support it.
* **Pinecone**: Pinecone support is coming soon. We'll add it to these benchmarks once available.
* **Faiss, Annoy, Scann**: We do not benchmark algorithms or ANN libraries. We only benchmark backends that can be used as 
Document stores. Actually we do not benchmark HNSW itself, but it is used by some backends internally.

## Conclusion

We hope our benchmark result can help users select the store that suits their use cases. Depending on the dataset size and the needed quality, some stores may be preferable than others. 

If you're experimenting on a dataset with fewer than 10,000 Documents, you can use the in-memory DocumentArray as-is to enjoy the best quality for nearest neighbor search with reasonable latency (say, less than 20 ms/query).

If your dataset does not fit in memory, and you **do not** care much about the speed of nearest neighbor search, you can use `sqlite` as storage.

When it comes to finding by vectors, the following figure shows the variation of latency w.r.t. different backend according to the dataset size. The red dashed line represents a time threshold of 0.3s, i.e. 300 ms:

```{figure} benchmark.svg
:scale: 125%
```

```{tip}
SQLite store is omitted from the left plot and the in-memory store is omitted from the right plot; because SQLite is too 
slow and the in-memory is too fast to fit into the figure.
```

AnnLite is a good choice when indexing/appending/inserting speed matters more than the speed of finding. Moreover, AnnLite is a local monolithic package that does not follow a client-server design, so it avoids all network overhead.

Weaviate and Qdrant offer the fastest approximate nearest neighbor search, while ElasticSearch offers a good trade-off between speed and quality. ElasticSearch performs the best in terms of quality of ANN, as we observed with highest Recall@K.
