import argparse
import functools
import random
from time import perf_counter

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from docarray import Document, DocumentArray
from rich.console import Console
from rich.table import Table

n_index_values = [10_000, 100_000, 1000_000]
n_query = 1
D = 128
TENSOR_SHAPE = (512, 256)
K = 10
n_vector_queries = 1000

parser = argparse.ArgumentParser()
parser.add_argument(
    '--default-hnsw',
    help='Whether to use default HNSW configurations',
    action='store_true',
)
args = parser.parse_args()

times = {}


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
def find_by_vector(da, query):
    return da.find(query, limit=K)


def get_docs(n, n_dim, tensor_shape, n_query):
    return [
        Document(
            embedding=np.random.rand(n_dim),
            # tensor=np.random.rand(*tensor_shape),
            tags={'i': int(i / n_query)},
        )
        for i in range(n)
    ]


def fmt(value, unit):
    return '{:.3f} {}'.format(value, unit)


def recall(predicted, relevant, eval_at):
    if eval_at == 0:
        return 0.0
    predicted_at_k = predicted[:eval_at]
    n_predicted_and_relevant = len(
        set(predicted_at_k[:, 'id']).intersection(set(relevant[:, 'id']))
    )
    return n_predicted_and_relevant / len(relevant)


if args.default_hnsw:
    storage_backends = [
        ('memory', None),
        ('qdrant', {'n_dim': D, 'scroll_batch_size': 8}),
        ('sqlite', None),
        (
            'annlite',
            {'n_dim': D},
        ),
        ('weaviate', {'n_dim': D}),
        ('elasticsearch', {'n_dim': D}),
    ]
else:
    storage_backends = [
        ('memory', None),
        ('qdrant', {'n_dim': D, 'scroll_batch_size': 8, 'ef_construct': 100, 'm': 16}),
        ('sqlite', None),
        (
            'annlite',
            {
                'n_dim': D,
                'ef_construction': 100,
                'ef_search': 100,
                'max_connection': 16,
            },
        ),
        (
            'weaviate',
            {'n_dim': D, 'ef': 100, 'ef_construction': 100, 'max_onnections': 16},
        ),
        ('elasticsearch', {'n_dim': D, 'ef_construction': 100, 'm': 16}),
    ]

table = Table(
    title=f'DocArray Benchmarking n_index={n_index_values[-1]} n_query={n_query} D={D} K={K}'
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
find_by_vector_values = {str(n_index): [] for n_index in n_index_values}
create_values = {str(n_index): [] for n_index in n_index_values}


for idx, n_index in enumerate(n_index_values):
    console.print(f'generating {n_index} docs...')
    docs = get_docs(n_index, D, TENSOR_SHAPE, n_query)
    docs_to_delete = random.sample(docs, n_query)
    docs_to_update = random.sample(docs, n_query)

    vector_queries = [np.random.rand(n_query, D) for _ in range(n_vector_queries)]
    ground_truth = []

    for backend, config in storage_backends:
        try:
            console.print('Backend:', backend.title())
            # for n_i in n_index:
            if not config:
                da = DocumentArray(storage=backend)
            else:
                da = DocumentArray(storage=backend, config=config)

            console.print(f'indexing {n_index} docs ...')
            create_time, _ = create(da, docs)

            # for n_q in n_query:
            console.print(f'reading {n_query} docs ...')
            read_time, _ = read(
                da,
                random.sample([d.id for d in docs], n_query),
            )

            console.print(f'updating {n_query} docs ...')
            update_time, _ = update(da, docs_to_update)

            console.print(f'deleting {n_query} docs ...')
            delete_time, _ = delete(da, [d.id for d in docs_to_delete])

            console.print(
                f'finding {n_query} docs by vector averaged {n_vector_queries} times ...'
            )
            if backend == 'sqlite':
                find_by_vector_time, result = find_by_vector(da, vector_queries[0])
                recall_at_k = recall(result, ground_truth[0], K)
            else:
                recall_at_k_values = []
                find_by_vector_times = []
                for i, query in enumerate(vector_queries):
                    find_by_vector_time, results = find_by_vector(da, query)
                    find_by_vector_times.append(find_by_vector_time)
                    if backend == 'memory':
                        ground_truth.append(results)
                        recall_at_k_values.append(1)
                    else:
                        recall_at_k_values.append(recall(results, ground_truth[i], K))

                recall_at_k = sum(recall_at_k_values) / len(recall_at_k_values)
                find_by_vector_time = sum(find_by_vector_times) / len(
                    find_by_vector_times
                )

            console.print(f'finding {n_query} docs by condition ...')
            find_by_condition_time, _ = find_by_condition(da, {'tags__i': {'$eq': 0}})

            if idx == len(n_index_values) - 1:
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
                benchmark_df.append(
                    pd.DataFrame(
                        [
                            [
                                backend.title(),
                                create_time,
                                read_time,
                                update_time,
                                delete_time,
                                find_by_vector_time,
                                recall_at_k,
                                find_by_condition_time,
                            ]
                        ],
                        columns=benchmark_df.columns,
                    )
                )

            find_by_vector_values[str(n_index)].append(find_by_vector_time)
            create_values[str(n_index)].append(create_time)
        except Exception as e:
            console.print(f'Storage Backend {backend} failed: {e}')

find_df = pd.DataFrame(find_by_vector_values)
find_df.index = [backend for backend, _ in storage_backends]
find_df = find_df.drop(['sqlite'])
print(find_df)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

find_df.plot(
    kind="bar",
    ax=ax1,
    color=sns.color_palette('muted')[1:4],
    title='Find by vector per backend and dataset size',
    ylabel='seconds',
    rot=0,
)

threshold = 0.3
ax1.hlines(y=threshold, xmin=-20, xmax=20, linewidth=2, color='r', linestyle='--')

create_df = pd.DataFrame(create_values)
create_df.index = [backend for backend, _ in storage_backends]

create_df = create_df.drop(['memory'])
print(create_df)

console.print(table)
plt.savefig('benchmark.svg')

benchmark_df.to_csv('benchmark.csv')
