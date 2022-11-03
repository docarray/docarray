# One Million Scale Benchmark

```{figure} https://docarray.jina.ai/_images/benchmark-banner.gif
:width: 0 %
:scale: 0 %
```

```{figure} benchmark-banner.gif
:scale: 0 %
```

We create a DocumentArray with one million Documents based on [SIFT1M](https://www.tensorflow.org/datasets/catalog/sift1m), a dataset containing 1 million objects of 128d and using l2 distance metrics, and benchmark the following document stores.

This includes classic database and vector database, all under the same DocumentArray API:

| Name                                | Usage                                    | Version           |
|-------------------------------------|------------------------------------------|-------------------|
| In-memory DocumentArray             | `DocumentArray()`                        | DocArray `0.18.2` |
| [`Weaviate`](weaviate.md)           | `DocumentArray(storage='weaviate')`      | `3.3.3`           |
| [`Qdrant`](qdrant.md)               | `DocumentArray(storage='qdrant')`        | `0.8.0`           |
| [`Annlite`](annlite.md)             | `DocumentArray(storage='anlite')`        | `0.3.13`          |
| [`ElasticSearch`](elasticsearch.md) | `DocumentArray(storage='elasticsearch')` | `8.4.3`           |
| [`Redis`](redis.md)                 | `DocumentArray(storage='redis')`         | `4.3.4`           |

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

The following chart and table summarize the result. The chart depicts Recall@10 (the fraction of true nearest neighbors found, on average over all queries) against QPS (find by vecotr queries per second). The smaller the time values and the more upper-right in the chart, the better.


## Charts

```{include} benchmark.html
```


````{tab} None

| max connections | ef construct |  ef  | Recall@10 | Find by vector (s) | Find by condition (s) | Create 1M (s) | Read (ms) | Update (ms) | Delete (ms) |
|-----------------|-------------:|-----:|----------:|-------------------:|----------------------:|--------------:|----------:|------------:|------------:|
|             N/A |          N/A |  N/A |     1.000 |               2.37 |                 11.17 |          1.06 |      0.17 |       0.05  |        0.14 |

````

````{tab} Annlite

| max connections | ef construct |  ef  | Recall@10 | Find by vector (ms) | Find by condition (ms) | Create 1M (s) | Read (ms) | Update (ms) | Delete (ms) |
|-----------------|-------------:|-----:|----------:|--------------------:|-----------------------:|--------------:|----------:|------------:|------------:|
|              16 |          128 |   64 |     0.960 |                1.53 |                   0.38 |        148.67 |      0.36 |       24.42 |       46.17 |
|              16 |          128 |  128 |     0.988 |                1.72 |                   0.36 |        130.47 |      0.34 |      210.10 |      227.95 |
|              16 |          128 |  256 |     0.996 |                1.99 |                   0.37 |        141.49 |      0.35 |      117.17 |      164.43 |
|              16 |          256 |   64 |     0.965 |                1.54 |                   0.37 |        186.36 |      0.36 |       32.40 |       45.42 |
|              16 |          256 |  128 |     0.990 |                1.69 |                   0.36 |        188.02 |      0.35 |       93.48 |      168.10 |
|              16 |          256 |  256 |     0.997 |                2.07 |                   0.36 |        183.66 |      0.36 |       18.86 |       35.82 |
|              32 |          128 |   64 |     0.975 |                1.58 |                   0.40 |        156.41 |      0.34 |       16.17 |       31.42 |
|              32 |          128 |  128 |     0.993 |                1.81 |                   0.37 |        147.05 |      0.35 |       19.81 |       39.87 |
|              32 |          128 |  256 |     0.998 |                2.15 |                   0.38 |        144.64 |      0.34 |       29.62 |       40.21 |
|              32 |          256 |   64 |     0.984 |                1.62 |                   0.37 |        211.81 |      0.35 |       32.65 |       35.15 |
|              32 |          256 |  128 |     0.996 |                1.84 |                   0.37 |        210.85 |      0.35 |      141.31 |      175.80 |
|              32 |          256 |  256 |     0.999 |                2.25 |                   0.37 |        204.65 |      0.35 |       22.13 |       31.54 |

````

````{tab} Qdrant

| max connections | ef construct |  ef  | Recall@10 | Find by vector (ms) | Find by condition (ms) | Create 1M (s) | Read (ms) | Update (ms) | Delete (ms) |
|-----------------|-------------:|-----:|----------:|--------------------:|-----------------------:|--------------:|----------:|------------:|------------:|
|              16 |           64 |   32 |     0.968 |               47.80 |                  44.96 |       5889.66 |      5.70 |       46.11 |       47.24 |
|              16 |           64 |   64 |     0.990 |               47.88 |                  44.72 |       5892.35 |      2.72 |       46.61 |       47.30 |
|              16 |           64 |  128 |     0.997 |               47.80 |                  44.95 |       5881.15 |      2.84 |       45.34 |       47.36 |
|              16 |          128 |   32 |     0.979 |               47.69 |                  44.67 |       5867.41 |      2.83 |       45.72 |       47.15 |
|              16 |          128 |   64 |     0.995 |               47.75 |                  44.68 |       5908.85 |      5.96 |       45.67 |       67.36 |
|              16 |          128 |  128 |     0.998 |               47.86 |                  43.96 |       6128.75 |      2.88 |       46.65 |       67.40 |
|              32 |           64 |   32 |     0.974 |               47.80 |                  44.76 |       5861.24 |      5.86 |       46.11 |       47.29 |
|              32 |           64 |   64 |     0.991 |               47.87 |                  43.90 |       5860.92 |      2.84 |       46.28 |       47.37 |
|              32 |           64 |  128 |     0.997 |               47.78 |                  43.51 |       5864.11 |      2.76 |       46.25 |       47.40 |
|              32 |          128 |   32 |     0.986 |               47.74 |                  44.59 |       5855.59 |      2.77 |       46.65 |       47.25 |
|              32 |          128 |   64 |     0.996 |               47.79 |                  43.75 |       5850.02 |      2.88 |       46.80 |       47.12 |
|              32 |          128 |  128 |     0.999 |               47.79 |                  43.77 |       5850.69 |      2.76 |       47.99 |       45.39 |

````

````{tab} Weaviate

| max connections | ef construct |  ef  | Recall@10 | Find by vector (ms) | Find by condition (ms) | Create 1M (s) | Read (ms) | Update (ms) | Delete (ms) |
|-----------------|-------------:|-----:|----------:|--------------------:|-----------------------:|--------------:|----------:|------------:|------------:|
|              16 |          128 |   64 |     0.959 |                5.29 |                   2.37 |       4847.50 |      4.96 |       27.86 |       17.70 |
|              16 |          128 |  128 |     0.988 |                5.84 |                   2.40 |       4647.66 |      2.99 |       16.32 |       18.06 |
|              16 |          128 |  256 |     0.996 |                6.91 |                   2.47 |       4551.60 |      2.77 |       27.50 |       18.22 |
|              16 |          256 |   64 |     0.965 |                5.31 |                   2.39 |       4794.40 |      2.84 |       17.99 |       18.18 |
|              16 |          256 |  128 |     0.990 |                5.89 |                   2.46 |       4600.66 |      2.86 |       18.03 |       18.14 |
|              16 |          256 |  256 |     0.998 |                6.85 |                   2.46 |       4511.55 |      2.93 |       17.58 |       17.69 |
|              32 |          128 |   64 |     0.975 |                5.52 |                   2.41 |       4520.41 |      2.86 |       16.56 |       17.85 |
|              32 |          128 |  128 |     0.993 |                6.16 |                   2.54 |       4453.26 |      2.85 |       26.57 |       18.18 |
|              32 |          128 |  256 |     0.998 |                7.45 |                   2.43 |       4406.78 |      2.86 |       28.64 |       18.31 |
|              32 |          256 |   64 |     0.984 |                5.62 |                   2.46 |       5060.14 |      2.93 |       18.97 |       17.82 |
|              32 |          256 |  128 |     0.996 |                6.34 |                   2.39 |       4846.96 |      5.00 |       21.84 |       26.85 |
|              32 |          256 |  256 |     0.999 |                8.17 |                   2.64 |       5019.82 |      2.84 |       22.97 |       29.82 |

````

````{tab} ElasticSearch

| max connections | ef construct |  ef  | Recall@10 | Find by vector (ms) | Find by condition (ms) | Create 1M (s) | Read (ms) | Update (ms) | Delete (ms) |
|-----------------|-------------:|-----:|----------:|--------------------:|-----------------------:|--------------:|----------:|------------:|------------:|
|              16 |          128 |   64 |     0.953 |                5.10 |                   6.64 |        678.95 |     15.25 |       47.03 |       43.08 |
|              16 |          128 |  128 |     0.981 |                6.43 |                   7.11 |        719.78 |     12.25 |       55.61 |       46.85 |
|              16 |          128 |  256 |     0.993 |                8.59 |                   7.01 |        720.77 |     16.59 |       64.65 |       58.07 |
|              16 |          256 |   64 |     0.958 |                5.43 |                   7.19 |       1138.32 |     18.90 |       73.47 |       62.13 |
|              16 |          256 |  128 |     0.983 |                6.60 |                   6.54 |       1078.97 |     11.58 |       73.65 |       56.86 |
|              16 |          256 |  256 |     0.993 |                8.80 |                   6.80 |       1108.34 |     12.93 |       60.73 |       47.59 |
|              32 |          128 |   64 |     0.984 |                5.72 |                   7.00 |        812.03 |     12.65 |       48.82 |       42.13 |
|              32 |          128 |  128 |     0.996 |                7.65 |                   7.46 |        861.62 |     12.32 |       61.79 |       57.73 |
|              32 |          128 |  256 |     0.999 |              1 0.44 |                   6.61 |        840.29 |     14.27 |       67.59 |       58.75 |
|              32 |          256 |   64 |     0.987 |                6.08 |                   7.51 |       1506.04 |     15.66 |       66.59 |       55.46 |
|              32 |          256 |  128 |     0.997 |                8.02 |                   6.63 |       1408.87 |     11.89 |       72.99 |       65.46 |
|              32 |          256 |  256 |     0.999 |              1 1.55 |                   7.69 |       1487.95 |     13.37 |       50.19 |       58.59 |

````

````{tab} Redis

| max connections | ef construct |  ef  | Recall@10 | Find by vector (ms) | Find by condition (ms) | Create 1M (s) | Read (ms) | Update (ms) | Delete (ms) |
|-----------------|-------------:|-----:|----------:|--------------------:|-----------------------:|--------------:|----------:|------------:|------------:|
|              16 |          128 |   64 |     0.959 |                1.78 |                   0.51 |        721.33 |      0.89 |        2.31 |       23.23 |
|              16 |          128 |  128 |     0.988 |                2.37 |                   0.70 |        775.26 |      1.24 |        4.25 |       28.60 |
|              16 |          128 |  256 |     0.997 |                2.63 |                   0.64 |        799.26 |      1.06 |        2.72 |       27.36 |
|              16 |          256 |   64 |     0.965 |                2.06 |                   0.66 |       1196.05 |      1.03 |        5.24 |       28.84 |
|              16 |          256 |  128 |     0.990 |                2.33 |                   0.62 |       1232.47 |      1.02 |        3.67 |       27.35 |
|              16 |          256 |  256 |     0.998 |                2.80 |                   0.67 |       1203.37 |      1.05 |        4.44 |       27.85 |
|              32 |          128 |   64 |     0.975 |                2.10 |                   0.67 |        953.10 |      1.06 |        2.35 |       27.11 |
|              32 |          128 |  128 |     0.993 |                2.49 |                   0.69 |        921.87 |      1.03 |        3.06 |       27.58 |
|              32 |          128 |  256 |     0.998 |                3.06 |                   0.64 |        926.96 |      1.06 |        2.45 |       27.27 |
|              32 |          256 |   64 |     0.984 |                2.28 |                   0.79 |       1489.83 |      1.05 |        4.92 |       29.27 |
|              32 |          256 |  128 |     0.996 |                2.75 |                   0.79 |       1511.17 |      1.05 |        4.03 |       28.48 |
|              32 |          256 |  256 |     0.999 |                3.15 |                   0.63 |       1534.68 |      1.03 |        3.26 |       28.19 |

````


When we consider each query as a Document, we can convert the above metrics into query/document per second, i.e. QPS/DPS. Values are higher the better (except for `Recall@10`). 

````{tab} None in QPS

| max connections | ef construct |  ef  | Recall@10 | Find by vector | Find by condition | Create 1M |  Read  | Update | Delete |
|-----------------|-------------:|-----:|----------:|---------------:|------------------:|----------:|-------:|-------:|-------:|
|             N/A |          N/A |  N/A |     1.000 |           0.42 |              0.09 |   947,284 |  6,061 | 21,505 |  7,246 |

````

````{tab} Annlite in QPS

| max connections | ef construct |  ef  | Recall@10 | Find by vector | Find by condition | Create 1M |  Read  | Update | Delete |
|-----------------|-------------:|-----:|----------:|---------------:|------------------:|----------:|-------:|-------:|-------:|
|              16 |          128 |   64 |     0.960 |            652 |             2,611 |     6,726 |  2,786 |     41 |     22 |
|              16 |          128 |  128 |     0.988 |            583 |             2,793 |     7,665 |  2,976 |      5 |      4 |
|              16 |          128 |  256 |     0.996 |            502 |             2,710 |     7,068 |  2,882 |      9 |      6 |
|              16 |          256 |   64 |     0.965 |            648 |             2,695 |     5,366 |  2,762 |     31 |     22 |
|              16 |          256 |  128 |     0.990 |            590 |             2,747 |     5,319 |  2,833 |     11 |      6 |
|              16 |          256 |  256 |     0.997 |            483 |             2,786 |     5,445 |  2,786 |     53 |     28 |
|              32 |          128 |   64 |     0.975 |            632 |             2,500 |     6,394 |  2,959 |     62 |     32 |
|              32 |          128 |  128 |     0.993 |            553 |             2,703 |     6,800 |  2,825 |     50 |     25 |
|              32 |          128 |  256 |     0.998 |            465 |             2,660 |     6,914 |  2,985 |     34 |     25 |
|              32 |          256 |   64 |     0.984 |            618 |             2,703 |     4,721 |  2,874 |     31 |     28 |
|              32 |          256 |  128 |     0.996 |            542 |             2,710 |     4,743 |  2,833 |      7 |      6 |
|              32 |          256 |  256 |     0.999 |            445 |             2,740 |     4,886 |  2,874 |     45 |     32 |

````

````{tab} Qdrant in QPS

| max connections | ef construct |  ef  | Recall@10 | Find by vector | Find by condition | Create 1M |  Read  | Update | Delete |
|-----------------|-------------:|-----:|----------:|---------------:|------------------:|----------:|-------:|-------:|-------:|
|              16 |           64 |   32 |     0.968 |             21 |                22 |       170 |    176 |     22 |     21 |
|              16 |           64 |   64 |     0.990 |             21 |                22 |       170 |    367 |     21 |     21 |
|              16 |           64 |  128 |     0.997 |             21 |                22 |       170 |    352 |     22 |     21 |
|              16 |          128 |   32 |     0.979 |             21 |                22 |       170 |    353 |     22 |     21 |
|              16 |          128 |   64 |     0.995 |             21 |                22 |       169 |    168 |     22 |     15 |
|              16 |          128 |  128 |     0.998 |             21 |                23 |       163 |    347 |     21 |     15 |
|              32 |           64 |   32 |     0.974 |             21 |                22 |       171 |    171 |     22 |     21 |
|              32 |           64 |   64 |     0.991 |             21 |                23 |       171 |    353 |     22 |     21 |
|              32 |           64 |  128 |     0.997 |             21 |                23 |       171 |    363 |     22 |     21 |
|              32 |          128 |   32 |     0.986 |             21 |                22 |       171 |    361 |     21 |     21 |
|              32 |          128 |   64 |     0.996 |             21 |                23 |       171 |    348 |     21 |     21 |
|              32 |          128 |  128 |     0.999 |             21 |                23 |       171 |    362 |     21 |     22 |

````

````{tab} Weaviate in QPS

| max connections | ef construct |  ef  | Recall@10 | Find by vector | Find by condition | Create 1M |  Read  | Update | Delete |
|-----------------|-------------:|-----:|----------:|---------------:|------------------:|----------:|-------:|-------:|-------:|
|              16 |          128 |   64 |     0.959 |            189 |               421 |       206 |    202 |     36 |     56 |
|              16 |          128 |  128 |     0.988 |            171 |               417 |       215 |    334 |     61 |     55 |
|              16 |          128 |  256 |     0.996 |            145 |               405 |       220 |    361 |     36 |     55 |
|              16 |          256 |   64 |     0.965 |            189 |               419 |       209 |    352 |     56 |     55 |
|              16 |          256 |  128 |     0.990 |            170 |               406 |       217 |    349 |     55 |     55 |
|              16 |          256 |  256 |     0.998 |            146 |               406 |       222 |    342 |     57 |     57 |
|              32 |          128 |   64 |     0.975 |            181 |               415 |       221 |    350 |     60 |     56 |
|              32 |          128 |  128 |     0.993 |            162 |               394 |       225 |    351 |     38 |     55 |
|              32 |          128 |  256 |     0.998 |            134 |               412 |       227 |    350 |     35 |     55 |
|              32 |          256 |   64 |     0.984 |            178 |               407 |       198 |    341 |     53 |     56 |
|              32 |          256 |  128 |     0.996 |            158 |               418 |       206 |    200 |     46 |     37 |
|              32 |          256 |  256 |     0.999 |            122 |               379 |       199 |    353 |     44 |     34 |

````

````{tab} ElasticSearch in QPS

| max connections | ef construct |  ef  | Recall@10 | Find by vector | Find by condition | Create 1M |  Read  | Update | Delete |
|-----------------|-------------:|-----:|----------:|---------------:|------------------:|----------:|-------:|-------:|-------:|
|              16 |          128 |   64 |     0.953 |            196 |               151 |     1,473 |     66 |     21 |     23 |
|              16 |          128 |  128 |     0.981 |            155 |               141 |     1,389 |     82 |     18 |     21 |
|              16 |          128 |  256 |     0.993 |            116 |               143 |     1,387 |     60 |     15 |     17 |
|              16 |          256 |   64 |     0.958 |            184 |               139 |       878 |     53 |     14 |     16 |
|              16 |          256 |  128 |     0.983 |            151 |               153 |       928 |     86 |     14 |     18 |
|              16 |          256 |  256 |     0.993 |            114 |               147 |       902 |     77 |     16 |     21 |
|              32 |          128 |   64 |     0.984 |            175 |               143 |     1,231 |     79 |     20 |     24 |
|              32 |          128 |  128 |     0.996 |            131 |               134 |     1,161 |     81 |     16 |     17 |
|              32 |          128 |  256 |     0.999 |             96 |               151 |     1,190 |     70 |     15 |     17 |
|              32 |          256 |   64 |     0.987 |            165 |               133 |       664 |     64 |     15 |     18 |
|              32 |          256 |  128 |     0.997 |            125 |               151 |       710 |     84 |     14 |     15 |
|              32 |          256 |  256 |     0.999 |             87 |               130 |       672 |     75 |     20 |     17 |

````


````{tab} Redis in QPS

| max connections | ef construct |  ef  | Recall@10 | Find by vector | Find by condition | Create 1M |  Read  | Update | Delete |
|-----------------|-------------:|-----:|----------:|---------------:|------------------:|----------:|-------:|-------:|-------:|
|              16 |          128 |   64 |     0.959 |            562 |             1,961 |     1,386 |  1,125 |    432 |     43 |
|              16 |          128 |  128 |     0.988 |            422 |             1,420 |     1,290 |    804 |    235 |     35 |
|              16 |          128 |  256 |     0.997 |            380 |             1,567 |     1,251 |    943 |    368 |     37 |
|              16 |          256 |   64 |     0.965 |            485 |             1,527 |       836 |    971 |    191 |     35 |
|              16 |          256 |  128 |     0.990 |            429 |             1,616 |       811 |    978 |    273 |     37 |
|              16 |          256 |  256 |     0.998 |            357 |             1,493 |       831 |    951 |    225 |     36 |
|              32 |          128 |   64 |     0.975 |            476 |             1,490 |     1,049 |    948 |    425 |     37 |
|              32 |          128 |  128 |     0.993 |            402 |             1,456 |     1,085 |    970 |    327 |     36 |
|              32 |          128 |  256 |     0.998 |            326 |             1,570 |     1,079 |    943 |    408 |     37 |
|              32 |          256 |   64 |     0.984 |            438 |             1,266 |       671 |    951 |    203 |     34 |
|              32 |          256 |  128 |     0.996 |            364 |             1,263 |       662 |    952 |    248 |     35 |
|              32 |          256 |  256 |     0.999 |            318 |             1,600 |       652 |    971 |    306 |     35 |

````

## Benchmark setup

We now elaborate the setup of our benchmark. First the following parameters are used:

| Parameter                                       | Value     |
|-------------------------------------------------|-----------|
| Number of created Documents                     | 1,000,000 |
| Number of Document on task 2,3,4,5,6            | 1         |
| The dimension of `.embedding`                   | 128       |
| Number of results for the task "Find by vector" | 10,000    |

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

The experiments were conducted on a AWS EC2 t2.2xlarge instance (Intel Xeon CPU E5-2676 v3 @ 2.40GHz) with Python 3.10.6 and DocArray 0.18.2.

Besides, as Weaviate, Qdrant, ElasticSearch and Redis follow a client/server pattern, we set up them with their official docker images in a **single node** configuration, with 32 GB of RAM allocated. That is, only 1 replica and shard are operated during the benchmarking. We did not opt for a cluster setup because our benchmarks mainly aim to assess the capabilities of a single instance of the server.

Results might include overhead coming from DocArray side which applies equally for all backends, unless a specific backend provides a more efficient implementation.

### Settings of the nearest neighbour search

As most of these document stores use their own implementation of HNSW (an approximate nearest neighbor search algorithm) but with different parameters:
1. ef_construct - The HNSW build parameter that controls the index time/index accuracy. Bigger `ef_construct` leads to longer construction, but better index quality.
2. max_connections - The number of bi-directional links created for every new element during construction. Higher `max_connections` work better on datasets with high intrinsic dimensionality and/or high recall, while low `max_connections` work better for datasets with low intrinsic dimensionality and/or low recall.
3. ef - The size of the dynamic list for the nearest neighbors. Higher `ef` at search leads to more accurate but slower search.

Finally, the full benchmark script is [available here](https://github.com/jina-ai/docarray/blob/main/scripts/benchmarking_sift1m.py).

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
same benchmarking experiment. Furthermore, we made sure to run the experiment with a series of HNSW parameters for backends that support 
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


```{tip}
SQLite store is omitted from the left plot and the in-memory store is omitted from the right plot; because SQLite is too 
slow and the in-memory is too fast to fit into the figure.
```

AnnLite is a good choice when indexing/appending/inserting speed matters more than the speed of finding. Moreover, AnnLite is a local monolithic package that does not follow a client-server design, so it avoids all network overhead.

