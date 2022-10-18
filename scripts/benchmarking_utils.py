import functools
import random
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from docarray import Document, DocumentArray
from rich.console import Console
from rich.table import Table


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        res = func(*args, **kwargs)
        return (perf_counter() - start, res)

    return wrapper


@timer
def create(da, docs):
    da.extend(docs)


@timer
def read(da, ids):
    da[ids]


@timer
def update(da, docs):
    da[[d.id for d in docs]] = docs


@timer
def delete(da, ids):
    del da[ids]


@timer
def find_by_condition(da, query):
    da.find(query)


@timer
def find_by_vector(da, query, limit, **kwargs):
    return da.find(query, limit=limit, **kwargs)


def get_docs(train):
    return [
        Document(
            embedding=x,
            tags={'i': int(i)},
        )
        for i, x in enumerate(train)
    ]


def fmt(value, unit):
    return '{:.3f} {}'.format(value, unit)


def get_configuration_storage_backends(argparse, D):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--default-hnsw',
        help='Whether to use default HNSW configurations',
        action='store_true',
    )

    parser.add_argument(
        '--exclude-backends',
        help='list of comma separated backends to exclude from the benchmarks',
        type=str,
    )

    args = parser.parse_args()

    if args.default_hnsw:
        storage_backends = [
            ('memory', None),
            ('sqlite', None),
            (
                'annlite',
                {
                    'n_dim': D,
                    'columns': {'i': 'int'},
                },
            ),
            (
                'qdrant',
                {
                    'n_dim': D,
                    'scroll_batch_size': 8,
                    'port': '41233',
                },
            ),
            (
                'weaviate',
                {
                    'n_dim': D,
                    'port': '41234',
                    'columns': {'i': 'int'},
                },
            ),
            (
                'elasticsearch',
                {
                    'n_dim': D,
                    'hosts': 'http://localhost:41235',
                    'columns': {'i': 'int'},
                    'es_config': {'timeout': 1000},
                },
            ),
            (
                'redis',
                {
                    'n_dim': D,
                    'port': '41236',
                    'distance': 'L2',
                    'columns': {'i': 'int'},
                },
            ),
        ]
    else:
        storage_backends = [
            ('memory', None),
            ('sqlite', None),
            (
                'annlite',
                {
                    'n_dim': D,
                    'ef_construction': 100,
                    'ef_search': 100,
                    'max_connection': 16,
                    'columns': {'i': 'int'},
                },
            ),
            (
                'qdrant',
                {
                    'n_dim': D,
                    'scroll_batch_size': 8,
                    'ef_construct': 100,
                    'm': 16,
                    'port': '41233',
                },
            ),
            (
                'weaviate',
                {
                    'n_dim': D,
                    'ef': 100,
                    'ef_construction': 100,
                    'max_connections': 16,
                    'port': '41234',
                    'columns': {'i': 'int'},
                },
            ),
            (
                'elasticsearch',
                {
                    'n_dim': D,
                    'ef_construction': 100,
                    'm': 16,
                    'hosts': 'http://localhost:41235',
                    'columns': {'i': 'int'},
                },
            ),
            (
                'redis',
                {
                    'n_dim': D,
                    'ef_construction': 100,
                    'm': 16,
                    'ef_runtime': 100,
                    'port': '41236',
                    'columns': {'i': 'int'},
                },
            ),
        ]

    storage_backends = [
        (backend, config)
        for backend, config in storage_backends
        if backend not in (args.exclude_backends or '').split(',')
    ]
    return storage_backends


storage_backend_filters = {
    'memory': {'tags__i': {'$eq': 0}},
    'sqlite': {'tags__i': {'$eq': 0}},
    'annlite': {'i': {'$eq': 0}},
    'qdrant': {'tags__i': {'$eq': 0}},
    'weaviate': {'path': 'i', 'operator': 'Equal', 'valueInt': 0},
    'elasticsearch': {'match': {'i': 0}},
    'redis': {'i': {'$eq': 0}},
}


def recall(predicted, relevant, eval_at):
    if eval_at == 0:
        return 0.0
    predicted_at_k = predicted[:eval_at]
    n_predicted_and_relevant = len(
        set(predicted_at_k[:, 'tags__i']).intersection(set(relevant))
    )
    return n_predicted_and_relevant / len(relevant)


def run_benchmark(
    train,
    test,
    ground_truth,
    n_index,
    n_vector_queries,
    n_query,
    storage_backends,
    K,
    D,
):
    table = Table(
        title=f'DocArray Benchmarking n_index={n_index} n_query={n_query} D={D} K={K}'
    )
    benchmark_df = pd.DataFrame(
        {
            'Storage Backend': [],
            'Indexing time (C)': [],
            'Query (R)': [],
            'Update (U)': [],
            'Delete (D)': [],
            'Find by vector': [],
            f'Recall at k={K} for vector search': [],
            'Find by condition': [],
        }
    )

    for col in benchmark_df.columns:
        table.add_column(col)

    console = Console()

    console.print(f'Reading dataset')
    docs = get_docs(train)
    docs_to_delete = random.sample(docs, n_query)
    docs_to_update = random.sample(docs, n_query)
    vector_queries = [x for x in test]

    find_by_vector_time_all = []
    create_time_all = []

    for backend, config in storage_backends:
        try:
            console.print('\nBackend:', backend.title())
            if not config:
                da = DocumentArray(storage=backend)
            else:
                da = DocumentArray(storage=backend, config=config)

            console.print(f'\tindexing {n_index} docs ...')
            create_time, _ = create(da, docs)

            # for n_q in n_query:
            console.print(f'\treading {n_query} docs ...')
            read_time, _ = read(
                da,
                random.sample([d.id for d in docs], n_query),
            )

            console.print(f'\tupdating {n_query} docs ...')
            update_time, _ = update(da, docs_to_update)

            console.print(f'\tdeleting {n_query} docs ...')
            delete_time, _ = delete(da, [d.id for d in docs_to_delete])

            console.print(
                f'\tfinding {n_query} docs by vector averaged {n_vector_queries} times ...'
            )

            if backend == 'memory' and len(ground_truth) == n_vector_queries:
                find_by_vector_time, results = find_by_vector(
                    da=da, query=vector_queries[0], limit=K, metric='euclidean'
                )
                recall_at_k = recall(results, ground_truth[0], K)
            elif backend == 'sqlite':
                find_by_vector_time, result = find_by_vector(
                    da, vector_queries[0], limit=K, metric='euclidean'
                )
                recall_at_k = recall(result, ground_truth[0], K)
            else:
                recall_at_k_values = []
                find_by_vector_times = []
                for i, query in enumerate(vector_queries):
                    find_by_vector_time, results = find_by_vector(da, query, limit=K)
                    find_by_vector_times.append(find_by_vector_time)
                    if backend == 'memory':
                        ground_truth.append(results[:, 'tags__i'])
                        recall_at_k_values.append(1)
                    else:
                        recall_at_k_values.append(recall(results, ground_truth[i], K))

                recall_at_k = np.mean(recall_at_k_values)
                find_by_vector_time = np.mean(find_by_vector_times)

            console.print(f'\tfinding {n_query} docs by condition ...')
            find_by_condition_time, _ = find_by_condition(
                da, storage_backend_filters[backend]
            )

            table.add_row(
                backend.title(),
                fmt(create_time, 's'),
                fmt(read_time * 1000, 'ms'),
                fmt(update_time * 1000, 'ms'),
                fmt(delete_time * 1000, 'ms'),
                fmt(find_by_vector_time, 's'),
                '{:.3f}'.format(recall_at_k),
                fmt(find_by_condition_time, 's'),
            )
            benchmark_df.loc[len(benchmark_df.index)] = [
                backend.title(),
                create_time,
                read_time,
                update_time,
                delete_time,
                find_by_vector_time,
                recall_at_k,
                find_by_condition_time,
            ]

            find_by_vector_time_all.append(find_by_vector_time)
            create_time_all.append(create_time)

            da.clear()
            del da
        except Exception as e:
            console.print(f'Storage Backend {backend} failed: {e}')

    # print(find_by_vector_time_all)
    console.print(table)
    return find_by_vector_time_all, create_time_all, benchmark_df


def save_benchmark_df(benchmark_df, n_index):
    benchmark_df.to_csv(f'benchmark-seconds-{n_index}.csv')

    benchmark_df['Indexing time (C)'] = benchmark_df['Indexing time (C)'].apply(
        lambda value: 1_000_000 / value
    )
    benchmark_df['Query (R)'] = benchmark_df['Query (R)'].apply(lambda value: 1 / value)
    benchmark_df['Update (U)'] = benchmark_df['Update (U)'].apply(
        lambda value: 1 / value
    )
    benchmark_df['Delete (D)'] = benchmark_df['Delete (D)'].apply(
        lambda value: 1 / value
    )
    benchmark_df['Find by vector'] = benchmark_df['Find by vector'].apply(
        lambda value: 1 / value
    )
    benchmark_df['Find by condition'] = benchmark_df['Find by condition'].apply(
        lambda value: 1 / value
    )

    benchmark_df.to_csv(f'benchmark-qps-{n_index}.csv')


def plot_results(
    find_by_vector_values, storage_backends, create_values, plot_legend=True
):
    find_df = pd.DataFrame(find_by_vector_values)
    find_df.index = [backend for backend, _ in storage_backends]
    find_df = find_df.drop(['sqlite'], errors='ignore')
    print('\n\nQuery times')
    print(find_df)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 5))

    find_df.plot(
        kind="bar",
        ax=ax1,
        fontsize=16,
        color=sns.color_palette('muted')[1:4],
        # title='Find by vector per backend and dataset size',
        # ylabel='seconds',
        rot=0,
        legend=plot_legend,
    )
    ax1.set_ylabel('seconds', fontsize=18)
    ax1.set_title('Find by vector per backend', fontsize=18)

    threshold = 0.3
    ax1.hlines(y=threshold, xmin=-20, xmax=20, linewidth=2, color='r', linestyle='--')

    create_df = pd.DataFrame(create_values)
    create_df.index = [backend for backend, _ in storage_backends]

    create_df = create_df.drop(['memory'])
    print('\n\nIndexing times')
    print(create_df)
    create_df.plot(
        kind="bar",
        ax=ax2,
        fontsize=16,
        color=sns.color_palette('muted')[1:4],
        # title='Indexing per backend and dataset size',
        # ylabel='seconds',
        rot=0,
        legend=plot_legend,
    )

    ax2.set_ylabel('seconds', fontsize=18)
    ax2.set_title('Indexing per backend', fontsize=18)

    plt.tight_layout()
    ax1.legend(fontsize=15)
    ax2.legend(fontsize=15)
    plt.savefig('benchmark.svg')
