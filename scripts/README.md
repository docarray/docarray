# Scripts

## Benchmarking on sift1m

### Download dataset

```shell
wget http://ann-benchmarks.com/sift-128-euclidean.hdf5
```

### Run storage backends

```shell
docker-compose up -d
```

### Run benchmarking script

Install dependencies:

```
pip install "docarray[common,benchmark,qdrant,annlite,weaviate,elasticsearch,redis]"
```

Run the benchmark on [sift1m](https://www.tensorflow.org/datasets/catalog/sift1m) or random dataset:

```
# sif1M
python benchmarking_sift1m.py [-h] [--exclude-backends EXCLUDE_BACKENDS] 

# random
python benchmarking.py [-h] [--fixed-hnsw] [--exclude-backends EXCLUDE_BACKENDS] 

options:
  -h, --help            show this help message and exit
  --fixed-hnsw          Whether to use default HNSW configurations for random dataset
  --exclude-backends EXCLUDE_BACKENDS
                        List of comma-separated backends to exclude from the benchmarks
```

Available storage backends include: `memory`, `sqlite`, `annlite`, `qdrant`, `weaviate`, `elasticsearch`, and `redis`.

The results of the benchmarks are stored in the current directory:
- `benchmark-seconds-{storage}.csv`
- `benchmark-qps-{n}.csv` 
- `benchmark.png`, a figure of of `Recall@10/QPS` for all storage backends


### Benchmark HNSW parameters

Current HNSW parameters are written in the function `get_configuration_storage_backends` of `benchmarking_utils.py`. Parameters include:

- **ef_construction/ef_construct**: Number of maximum allowed potential outgoing edges candidates for each node in the graph, during the graph building. 
- **max_connection/max_connections/m**: Number of maximum allowed outgoing edges for each node in the graph in each layer.
- **ef_search/hnsw_ef/ef/num_candidates/ef_runtime**: Number of maximum top candidates to hold during the KNN search.



