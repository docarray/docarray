import argparse

import numpy as np

from benchmarking_utils import (
    get_configuration_storage_backends,
    plot_results,
    run_benchmark,
    save_benchmark_df,
)

if __name__ == "__main__":

    # Parameters settable by the user
    n_index_values = [1_000_000]
    n_query = 1
    D = 128
    TENSOR_SHAPE = (512, 256)
    K = 10
    n_vector_queries = 1000
    np.random.seed(123)

    # Benchmark
    storage_backends = get_configuration_storage_backends(argparse, D)
    find_by_vector_values = {str(n_index): [] for n_index in n_index_values}
    create_values = {str(n_index): [] for n_index in n_index_values}

    for idx, n_index in enumerate(n_index_values):
        train = [np.random.rand(D) for _ in range(n_index)]
        test = [np.random.rand(D) for _ in range(n_vector_queries)]
        ground_truth = []

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
        find_by_vector_values[str(n_index)] = find_by_vector_time_all
        create_values[str(n_index)] = create_time_all
        save_benchmark_df(benchmark_df, n_index)

    plot_results(
        find_by_vector_values, storage_backends, create_values, plot_legend=False
    )
