import h5py

from benchmarking_utils import (
    run_benchmark2,
    save_benchmark_df,
)

if __name__ == "__main__":

    # Parameters settable by the user
    n_query = 1
    K = 10
    DATASET_PATH = 'sift-128-euclidean.hdf5'

    # Variables gathered from the dataset
    dataset = h5py.File(DATASET_PATH, 'r')
    train = dataset['train'][:1000]
    test = dataset['test'][:100]
    D = train.shape[1]
    n_index = len(train)
    n_vector_queries = len(test)
    ground_truth = [x[0:K] for x in dataset['neighbors'][0:n_vector_queries]]

    # Benchmark
    storage = 'redis'
    storage_config = [
        {
            'n_dim': D,
            'port': '41236',
            'distance': 'L2',
            'columns': {'i': 'int'},
            'm': 16,
            'ef_construction': 200,
            'ef_runtime': 10,
        },
        {
            'n_dim': D,
            'port': '41236',
            'distance': 'L2',
            'columns': {'i': 'int'},
            'm': 16,
            'ef_construction': 200,
            'ef_runtime': 20,
        },
        {
            'n_dim': D,
            'port': '41236',
            'distance': 'L2',
            'columns': {'i': 'int'},
            'm': 16,
            'ef_construction': 400,
            'ef_runtime': 10,
        },
    ]
    benchmark_df = run_benchmark2(
        train,
        test,
        ground_truth,
        n_index,
        n_vector_queries,
        n_query,
        storage,
        storage_config,
        K,
    )

    # store benchmark time
    save_benchmark_df(benchmark_df, storage)
