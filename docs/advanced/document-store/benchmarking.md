# Benchmarking
## Results

The script `scripts/benchmarking.py` benchmarks DocArray's supported Document Stores in 6 different operations:
* Create (indexing Documents)
* Read
* Update
* Delete
* Find Document by vector (Nearest Neighbor Search or Approximate Nearest Neighbor Search depending on the `backend`)
* Find Document by condition (apply filter)

````{tab} #1: Same HNSW parameters

| Backend       | Create (s) | Read (ms) | Update (ms) | Delete (ms) | Find by vector (s) | Recall at k=10 for | Find by condition (s) |
|---------------|------------|-----------|-------------|-------------|--------------------|--------------------|-----------------------|
| Memory        | 0.636      | 0.118     | 0.013       | 0.344       | 1.424              | 1.000              | 5.206                 |
| Sqlite        | 4236.665   | 0.333     | 0.312       | 24.936      | 21.420             | 1.000              | 30.015                |
| Annlite       | 72.121     | 0.246     | 8.723       | 3.552       | 0.432              | 0.188              | 29.914                |
| Weaviate      | 1601.777   | 2.002     | 34.111      | 22.841      | 0.005              | 0.110              | 1169.032              |
| ElasticSearch | 684.825    | 10.733    | 35.358      | 54.208      | 0.246              | 0.856              | 652.541               |

````

````{tab} #2: Default HNSW parameters

| Backend       | Create (s) | Read (ms) | Update (ms) | Delete (ms) | Find by vector (s) | Recall at k=10 for | Find by condition (s) |
|---------------|------------|-----------|-------------|-------------|--------------------|--------------------|-----------------------|
| Memory        | 0.636      | 0.118     | 0.013       | 0.344       | 1.424              | 1.000              | 5.206                 |
| Sqlite        | 4236.665   | 0.333     | 0.312       | 24.936      | 21.420             | 1.000              | 30.015                |
| Annlite       | 72.121     | 0.246     | 8.723       | 3.552       | 0.432              | 0.188              | 29.914                |
| Weaviate      | 1601.777   | 2.002     | 34.111      | 22.841      | 0.005              | 0.110              | 1169.032              |
| ElasticSearch | 684.825    | 10.733    | 35.358      | 54.208      | 0.246              | 0.856              | 652.541               |

````

## Experiment Settings

Since most of these Document Stores use their implementation of the HNSW Approximate Nearest Neighbor Search algorithm, 
with various default HNSW parameters, we conducted 2 benchmarking experiments for the `Find By Vector` operation:
1. Set up the Document Stores with the same HNSW parameters 
2. Set up the Document Stores with their default HNSW parameters at the time the benchmarking experiment was conducted.

Furthermore, we provide the `Recall At K` value, considering the exhaustive search as the ground truth. This allows 
you to also take into consideration the quality, not just the speed.

```{important}
The Sqlite and the in-memory Document Stores ** do not implement ** approximate nearest neighbor search and offer 
exhaustive search instead. That's why, they give the maximum quality but are the slowest when it comes to search nearest 
neighbors.
```

The results were conducted on a 4.5 Ghz AMD Ryzen Threadripper 3960X 24-Core Processor with Python 3.8.5 and using the official docker 
images of the storage backends. The docker images were allocated 40 GB of RAM.

The benchmarking experiments used the following parameters:
* Number of indexed Documents: 1M
* Number of query Documents: 1
* Embedding dimension: 128
* Number of Documents (K) to be retrieved for `Find Document by vector`: 10

For the first experiment, we fixed the following HNSW parameters for the Document Stores that support ANN:
* `ef` (construction): 100
* `ef` (search): 100
* `m` (max connections): 16

## Rationale
Our experiments are designed to be fair and the same across all backends while favouring Document Stores that benefit 
the DocArray user. Therefore, we based our benchmarks on the following:

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
GRPC instead of HTTP, use batch operations,... And we're okay with that, as long as it benefits the DocArray and Jina 
user.
* Open to improvements: We are constantly improving the performance of storage backends from the DocArray side and 
updating the benchmarks accordingly. If you believe we missed an optimization (perform an operation in batches, benefit 
from a recent feature in upstream, avoid unnecessary steps,...), feel free to raise a PR or issue, we're open to 
your contributions!

## Backends covered
Our benchmarks cover all backends supported by DocArray:
* In-memory
* Sqlite
* Qdrant
* Annlite
* Weaviate
* ElasticSearch

We do not cover the following backends for those reasons:
* Milvus: currently DocArray does not integrate with Milvus. We're open for contributions to DocArray's repository to 
support it.
* Pinecone: Pinecode support is coming soon, and we'll add it to these benchmarks once available.
* Faiss, Annoy, Scann: We do not benchmark algorithms or ANN libraries. We only benchmark backends that can be used as 
Document stores. Actually we do not benchmark HNSW itself, but it is used by some backends internally.


## Conclusion
To conclude, you can look at those benchmarks and select the backend that suits better your case.
If you're playing around on a small dataset, you can use `memory` as storage.
If you don't care much about the Nearest Neighbor Search speed and you're experiment with larger data, you can use`
sqlite` as storage.

