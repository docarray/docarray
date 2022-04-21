# One Million Scale Benchmark


```{figure} https://docarray.jina.ai/_images/benchmark-banner.gif
:width: 0 %
:scale: 0 %
```

```{figure} benchmark-banner.gif
:scale: 0 %
```

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

| Store         | Create (s) | Read (ms) | Update (ms) | Delete (ms) | Find by condition (s) | Find by vector (s) | Recall@10 |
|---------------|-----------:|----------:|------------:|------------:|----------------------:|-------------------:|----------:|
| Memory        |  ** 0.6 ** | ** 0.1 ** |  ** 0.01 ** |  ** 0.39 ** |               ** 5 ** |               1.43 |      1.00 |
| Sqlite        |    4,366.8 |       0.3 |        0.35 |        2.62 |                    31 |              23.59 |      1.00 |
| Annlite       |       72.4 |       0.3 |        6.57 |        4.62 |                    30 |               0.42 |      0.19 |
| Qdrant        |    2,286.3 |       1.6 |        1.50 |        4.14 |                   605 |         ** 0.01 ** |      0.51 |
| Weaviate      |    1,612.4 |      33.7 |       27.94 |       13.38 |                 1,242 |               0.02 |      0.11 |
| ElasticSearch |    1,307.3 |       2.4 |       40.85 |       40.74 |                   656 |               0.23 |  **0.85** |

````


````{tab} Same HNSW parameters (QPS)

| Store         |          Create |            Read |           Update |          Delete | Find by condition | Find by vector | Recall@10 |
|---------------|----------------:|----------------:|-----------------:|----------------:|------------------:|---------------:|----------:|
| Memory        | ** 1,610,305 ** |     ** 9,345 ** | ** 71,428.571 ** |     ** 2,570 ** |       ** 0.190 ** |          0.697 |      1.00 |
| Sqlite        |              22 |            3125 |            2,816 |             381 |             0.032 |          0.042 |      1.00 |
| Annlite       |           1,380 |           4,000 |              152 |             216 |             0.033 |          2.336 |      0.19 |
| Qdrant        |              43 |             604 |              664 |             241 |             0.002 |   ** 90.909 ** |      0.51 |
| Weaviate      |              62 |              31 |               35 |              74 |             0.001 |         55.556 |      0.11 |
| ElasticSearch |              76 |             409 |               24 |              24 |             0.002 |          4.386 |  **0.85** |

````

````{tab} Default HNSW parameters

| Backend       | Create (s) | Read (ms) | Update (ms) | Delete (ms) | Find by condition (s) | Find by vector (s) | Recall@10 |
|---------------|-----------:|----------:|------------:|------------:|----------------------:|-------------------:|----------:|
| Memory        |  ** 0.6 ** | ** 0.1 ** |  ** 0.01 ** |  ** 0.16 ** |               ** 5 ** |               1.43 |      1.00 |
| Sqlite        |    4,446.6 |       0.3 |        0.35 |       16.77 |                    30 |              24.38 |      1.00 |
| Annlite       |      114.0 |       0.3 |        9.36 |       20.09 |                    30 |               0.43 |      0.14 |
| Qdrant        |    2,227.4 |       1.6 |       42.16 |       20.59 |                   608 |         ** 0.01 ** |      0.51 |
| Weaviate      |    1,612.0 |       2.3 |       44.01 |       22.26 |                 1,208 |         ** 0.01 ** |      0.10 |
| ElasticSearch |      715.2 |       2.1 |       15.58 |       33.26 |                   650 |               0.22 |  **0.83** |

````

````{tab} Default HNSW parameters (QPS)

| Store         |          Create |            Read |           Update |         Delete | Find by condition | Find by vector | Recall@10 |
|---------------|----------------:|----------------:|-----------------:|---------------:|------------------:|---------------:|----------:|
| Memory        | ** 1,618,122 ** |    ** 10,101 ** |     ** 71,428 ** |    ** 6,211 ** |       ** 0.192 ** |          0.699 |      1.00 |
| Sqlite        |             224 |           3,125 |            2,824 |             59 |             0.033 |          0.041 |      1.00 |
| Annlite       |           8,769 |           3,759 |              106 |             49 |             0.033 |          2.347 |      0.14 |
| Qdrant        |             448 |             615 |               23 |             48 |             0.002 |  ** 111.111 ** |      0.51 |
| Weaviate      |             620 |             435 |               22 |             44 |             0.001 |        200.000 |      0.10 |
| ElasticSearch |           1,398 |             481 |              64. |             30 |             0.002 |          4.525 |  **0.83** |

````

## Benchmark setup

We now elaborate the setup of our benchmark. The benchmarking experiments used the following parameters:

| Parameter                                         | Value     |
|---------------------------------------------------|-----------|
| Number of created Documents                       | 1,000,000 |
| Number of Document on task 2,3,4,5,6              | 1         |
| The dimension of `.embedding`                     | 128       |
| Number of Documents for the task "Find by vector" | 10        |

We use the `Recall@K` value as an indicator of the search quality. The in-memory and SQLite store **do not implement** approximate nearest neighbor search but use exhaustive search instead. Hence, they give the maximum `Recall@K` but are the slowest. 

We chose to index 1 million Documents in these benchmarks since it's the breaking point at which Approximate Nearest 
Neighbor Search algorithms outperform naive approaches for production systems.

The experiments were conducted on a 4.5 Ghz AMD Ryzen Threadripper 3960X 24-Core Processor with Python 3.8.5.

Besides, since Weaviate, Qdrant and ElasticSearch follow a Client/Server pattern, they are set up in their official 
docker images (with 40 GB of RAM allocated) in a **single node** configuration. That is, only 1 replica and shard is 
operated during the benchmarking. We did not opt for a cluster setup because our benchmarks mainly aim to assess the 
capabilities of a single instance of the backend.

The indexed Documents follow the structure below:
```json
{
  "id": "94ee6627ee7f582e5e28124e78c3d2f9",
  "tags": {"i": 10},
  "embedding": [0.49841760378680844, 0.703959752118305, 0.6920759535687985, 0.10248648858410625, 0.39724992529516323, 0.9525701623588285, 0.6123678207411607, 0.8235835556434232, 0.1334310479203321, 0.019069168045542484, 0.46563554718054745, 0.3810651579771044, 0.8158992017970917, 0.1867398342916442, 0.36399925593012894, 0.8543925510346941, 0.9704327023477919, 0.9510717714842882, 0.10112728212884692, 0.23857235847278668, 0.46203675687572876, 0.8709349745561471, 0.2919425692798532, 0.1270317128569054, 0.598267403848796, 0.5079556629000722, 0.6723370767610658, 0.7499190013798289, 0.4071757704139839, 0.3266601616105066, 0.38251666657923256, 0.2641338590111115, 0.5818284950519711, 0.08409247398152986, 0.9431400762354458, 0.16266071169130447, 0.11449753491515469, 0.776299638603005, 0.483583028399195, 0.3495214545281775, 0.14102320650300604, 0.32028026696917755, 0.2328207668422747, 0.8110754637500887, 0.3289243490400434, 0.7966120143110351, 0.3131652505343142, 0.5194599974478354, 0.5689714910888303, 0.49354571385944135, 0.36647934405376115, 0.7014887661283724, 0.20642861485586006, 0.9402395370732294, 0.7745510547369284, 0.8629215543566361, 0.9133059632853437, 0.698747844103704, 0.32801918090207294, 0.6631657916961102, 0.4569921649166002, 0.31714923552973173, 0.22310861030947393, 0.38821853664940253, 0.6572612700653918, 0.49174075029773967, 0.5849767677338287, 0.21753609191853263, 0.47427937362536443, 0.20553977938157697, 0.021650687776025968, 0.552938630052999, 0.10481406498805323, 0.07384979186002205, 0.5303044812193785, 0.9930043152649674, 0.2123623783543509, 0.15978330380676908, 0.48928966291799203, 0.27449587583130663, 0.8746990697209669, 0.962953308417367, 0.0008612083303587426, 0.14060396178519263, 0.20409286324776232, 0.3504328802039508, 0.950473483991325, 0.22058248402077796, 0.6424150490118353, 0.2962973966442942, 0.4610734132578128, 0.3350055813069074, 0.1227227671592317, 0.5344683047388401, 0.713857955225115, 0.08368901111433469, 0.3599521978718555, 0.9406651190040148, 0.39743591295592273, 0.6236339280928341, 0.2175065637031257, 0.7163032467853059, 0.17699742657001782, 0.22577550530298418, 0.946237528694237, 0.6186134469700066, 0.9887108036340337, 0.5603507404807759, 0.4260786910866563, 0.18211790663973826, 0.35356283818138, 0.7364362209206111, 0.44741826983117916, 0.9644167469652952, 0.032334460943219234, 0.5813174431764855, 0.9521668648235565, 0.00225501952136975, 0.13570821521764775, 0.5249644506152132, 0.31344447815202425, 0.8600354853228754, 0.3855065571988878, 0.4966344954830819, 0.7192262653602179, 0.7871124456461347, 0.5592901097248714, 0.030080902555281064]
}
```

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

We hope our benchmark result can help user select the backend that suits better to its use case. Depending on the dataset size and the needed quality, some backends may be preferable  than others. 

If you're playing around on a dataset with less than 10k Documents, you can use default DocumentArray as-is (i.e. `memory` as storage) to enjoy the best 
quality for nearest neighbor search with reasonable time latency (less than 20 ms).

If your dataset does not fit in memory, and you **do not** care much about the speed of nearest neighbor search, you can use
`sqlite` as storage.


When it comes to finding by vectors, the following figure 
shows the variation of the time latency w.r.t. different backend according to the dataset size. The red dashed line represents a time threshold of 0.3s, i.e. 300 ms:

```{figure} benchmark.svg
:scale: 125%
```

```{tip}
Sqlite backend is omitted from the first plot and the memory backend is omitted from the second plot because one is too 
slow for vector retrieval and the other is too fast for indexing.
```

AnnLite is a good choice when indexing/appending/inserting speed matters more than the speed of finding by vectors. As AnnLite is a local monolith package that does not follow a client-server design, hence it avoids all network overhead.

Weaviate and Qdrant offer the fastest approximate nearest neighbor search, while ElasticSearch offers a good trade-off between speed and quality. ElasticSearch performs the best on the quality of ANN as we observed with highest Recall@K.
