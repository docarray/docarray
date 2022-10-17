import argparse

import h5py

from benchmarking_utils import (
    get_configuration_storage_backends,
    plot_results,
    run_benchmark,
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

    # Benchmark
    storage_backends = get_configuration_storage_backends(argparse, D)
    find_by_vector_time_all, create_time_all, benchmark_df = run_benchmark(
        train,
        test,
        ground_truth,
        n_index,
        n_vector_queries,
        n_query,
        storage_backends,
        K,
        D,
    )

    # store find_by_vector time
    find_by_vector_values = {str(n_index): find_by_vector_time_all}
    create_values = {str(n_index): create_time_all}
    save_benchmark_df(benchmark_df, n_index)

    plot_results(
        find_by_vector_values, storage_backends, create_values, plot_legend=False
    )
