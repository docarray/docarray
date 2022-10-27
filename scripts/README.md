# Scripts

## Benchmarking on sift1M

### Download dataset

```shell
wget http://ann-benchmarks.com/sift-128-euclidean.hdf5
```

### Run storage backends

```shell
docker-compose up -d
```

### Run benchmarking script

Run the benchmark:

```
python benchmarking_sift1m.py [-h] [--fixed-hnsw] [--exclude-backends EXCLUDE_BACKENDS]

options:
  -h, --help            show this help message and exit
  --fixed-hnsw          Whether to use default HNSW configurations for random dataset
  --exclude-backends EXCLUDE_BACKENDS
                        List of comma-separated backends to exclude from the benchmarks
```

Available storage backends include: `memory`, `sqlite`, `annlite`, `qdrant`, `weaviate`, `elasticsearch`, and `redis`.

Results of the benchmarks are stored in current directory. Results are csv files `benchmark-seconds-{storage}.csv` and `benchmark-qps-{n}.csv` and a figure `benchmark.png` of `Recall@10/QPS` for all storage backends.


### Benchmark HNSW parameters

Current HNSW parameters are written in function `get_configuration_storage_backends` of `benchmarking_utils.py`. Parameters include:

- **ef_construction/ef_construct**: Number of maximum allowed potential outgoing edges candidates for each node in the graph, during the graph building. 
- **max_connection/max_connections/m**: Number of maximum allowed outgoing edges for each node in the graph in each layer.
- **ef_search/hnsw_ef/ef/num_candidates/ef_runtime**: Number of maximum top candidates to hold during the KNN search.



