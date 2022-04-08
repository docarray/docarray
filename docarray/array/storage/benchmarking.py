import functools
import random
from time import perf_counter

import numpy as np
from docarray import Document, DocumentArray
from rich.console import Console
from rich.table import Table

n_index = 1_000_000
# n_index = [10_000, 100_000, 500_000, 1_000_000]
n_query = 1
# n_query = [1, 8, 64]
D = 128
TENSOR_SHAPE = (512, 256)


times = {}


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        func(*args, **kwargs)
        return perf_counter() - start

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
    da.find(query)


def get_docs(n, n_dim, tensor_shape, n_query):
    return [
        Document(
            embedding=np.random.rand(n_dim),
            # tensor=np.random.rand(*tensor_shape),
            tags={'i': int(i / n_query)},
        )
        for i in range(n)
    ]


def fmt(value):
    return "%.5f s" % value


storage_backends = [
    ('qdrant', {'n_dim': D, 'scroll_batch_size': 8}),
    ('memory', None),
    ('sqlite', None),
    ('annlite', {'n_dim': D}),
    ('weaviate', {'n_dim': D}),
    ('elasticsearch', {'n_dim': D}),
]
table = Table(title=f"DocArray Benchmarking n_index={n_index} n_query={n_query} D={D}")
table.add_column("Storage Backend")
table.add_column("Indexing time (C)")
table.add_column("Query (R)")
table.add_column("Update (U)")
table.add_column("Delete (D)")
table.add_column("Find by vector")
table.add_column("Find by condition")

for backend, config in storage_backends:
    print('Backend:', backend.title())
    # for n_i in n_index:
    if not config:
        da = DocumentArray(storage=backend)
    else:
        da = DocumentArray(storage=backend, config=config)

    print(f'generating {n_index} docs...')
    docs = get_docs(n_index, D, TENSOR_SHAPE, n_query)

    print(f'indexing {n_index} docs ...')
    create_time = create(da, docs)

    # for n_q in n_query:
    print(f'reading {n_query} docs ...')
    read_time = read(
        da,
        random.sample([d.id for d in docs], n_query),
    )

    print(f'updating {n_query} docs ...')
    update_time = update(da, random.sample(docs, n_query))

    docs_to_delete = random.sample(docs, n_query)
    print(f'deleting {n_query} docs ...')
    delete_time = delete(da, [d.id for d in docs_to_delete])

    print(f'finding {n_query} docs by vector ...')
    find_by_vector_time = find_by_vector(da, np.random.rand(n_query, D))

    print(f'finding {n_query} docs by condition ...')
    find_by_condition_time = find_by_condition(da, {'tags__i': {'$eq': 0}})

    table.add_row(
        backend.title(),
        fmt(create_time),
        fmt(read_time),
        fmt(update_time),
        fmt(delete_time),
        fmt(find_by_vector_time),
        fmt(find_by_condition_time),
    )

console = Console()
console.print(table)
