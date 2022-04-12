import functools
import random
from time import perf_counter

import numpy as np
from docarray import Document, DocumentArray
from rich.console import Console
from rich.table import Table

n_index = 1000_000
n_query = 1
D = 128
TENSOR_SHAPE = (512, 256)
K = 10
n_vector_queries = 1000


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


storage_backends = [
    ('memory', None),
    ('qdrant', {'n_dim': D, 'scroll_batch_size': 8, 'ef_construct': 100, 'm': 16}),
    ('sqlite', None),
    (
        'annlite',
        {'n_dim': D, 'ef_construction': 100, 'ef_search': 100, 'max_connection': 16},
    ),
    ('weaviate', {'n_dim': D, 'ef': 100, 'ef_construction': 100, 'max_onnections': 16}),
    ('elasticsearch', {'n_dim': D, 'ef_construction': 100, 'm': 16}),
]
table = Table(
    title=f"DocArray Benchmarking n_index={n_index} n_query={n_query} D={D} K={K}"
)
table.add_column("Storage Backend")
table.add_column("Indexing time (C)")
table.add_column("Query (R)")
table.add_column("Update (U)")
table.add_column("Delete (D)")
table.add_column("Find by vector")
table.add_column(f"Recall at k={K} for vector search")
table.add_column("Find by condition")

console = Console()

console.print(f'generating {n_index} docs...')
docs = get_docs(n_index, D, TENSOR_SHAPE, n_query)

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
        update_time, _ = update(da, random.sample(docs, n_query))

        docs_to_delete = random.sample(docs, n_query)
        console.print(f'deleting {n_query} docs ...')
        delete_time, _ = delete(da, [d.id for d in docs_to_delete])

        console.print(
            f'finding {n_query} docs by vector averaged {n_vector_queries} times ...'
        )
        if backend == 'sqlite':
            find_by_vector_time, _ = find_by_vector(da, query)
            recall_at_k = 1
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
            find_by_vector_time = sum(find_by_vector_times) / len(find_by_vector_times)

        console.print(f'finding {n_query} docs by condition ...')
        find_by_condition_time, _ = find_by_condition(da, {'tags__i': {'$eq': 0}})

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
    except Exception as e:
        console.print(f'Storage Backend {backend} failed: {e}')

console.print(table)
