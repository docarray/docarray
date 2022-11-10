import argparse
from itertools import product

import h5py

from benchmarking_utils import (
    get_configuration_storage_backends,
    get_docs,
    plot_results_sift,
    run_benchmark_sift,
    save_benchmark_df,
)

if __name__ == "__main__":

    # Parameters settable by the user
    n_query = 1
    K = 10
    DATASET_PATH = 'sift-128-euclidean.hdf5'

    # Variables gathered from the dataset
    dataset = h5py.File(DATASET_PATH, 'r')
    train = dataset['train']
    test = dataset['test']
    D = train.shape[1]
    n_index = len(train)
    n_vector_queries = len(test)
    ground_truth = [x[0:K] for x in dataset['neighbors'][0:n_vector_queries]]

    BENCHMARK_CONFIG = get_configuration_storage_backends(argparse, D, False)

    print(f'Reading dataset')
    docs = get_docs(train)

    for storage, cfg in BENCHMARK_CONFIG.items():

        storage_config = cfg['storage_config']
        hnsw_config = []

        if storage != 'memory' and storage != 'sqlite':
            for hnsw_cfg in product(*cfg['hnsw_config'].values()):
                hnsw_config.append(dict(zip(cfg['hnsw_config'].keys(), hnsw_cfg)))
        else:
            hnsw_config.append({})

        benchmark_df = run_benchmark_sift(
            test,
            docs,
            ground_truth,
            n_index,
            n_vector_queries,
            n_query,
            storage,
            storage_config,
            hnsw_config,
            K,
        )

        # store benchmark time
        save_benchmark_df(benchmark_df, storage)

    plot_results_sift(BENCHMARK_CONFIG.keys())
