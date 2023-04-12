# Store

If you work with multi-modal data, usually you want to **store** it somewhere.

DocArray offers to ways of storing your data:

1. In a **[Document Index](#document-index)** for fast retrieval using vector similarity
2. In a **[Document Store](#document-store)** for simple long-term storage

## Document Index

A Document Index lets you store your Documents and search through them using vector similarity.

This is useful if you want to store a bunch of data, and at a later point retrieve Documents that are similar to
some query that you provide.
Concrete examples where this is relevant are neural search application, Augmenting LLMs and Chatbots with domain knowledge ([Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401))]),
or recommender systems.

!!! question "How does vector similarity search work?"
    TODO

DocArray's Document Index concept achieves this by providing a unified interface to a number of [vector databases](https://learn.microsoft.com/en-us/semantic-kernel/concepts-ai/vectordb).
In fact, you can think of Document Index as an **[ORM](https://sqlmodel.tiangolo.com/db-to-code/) for vector databases**.

Currently, DocArray supports the following vector databases:
- [Weaviate](https://weaviate.io/)  |  [Docs](TODO)
- [Qdrant](https://qdrant.tech/)  |  [Docs](TODO)
- [Elasticsearch](https://www.elastic.co/elasticsearch/)  |  [Docs v8](TODO), [Docs v7](TODO)
- [HNSWlib](https://github.com/nmslib/hnswlib)  |  [Docs](TODO)

For this user guide you will use the [HNSWLibDocumentIndex](docarray.index.backends.hnswlib.HnswDocumentIndex)
because it doesn't require you to launch a database server. Instead, it will store your data locally.

!!! note "Using a different vector database"
    You can easily use Weaviate, Qdrant, or Elasticsearch instead, they share the same API!
    To do so, check out their respective documentation sections.

!!! note "HNSWLib-specific settings"
    The following sections explain the general concept of Document Index by using
    [HNSWLibDocumentIndex](docarray.index.backends.hnswlib.HnswDocumentIndex) as an example.
    For HNSWLib-specific settings, check out the [HNSWLibDocumentIndex](docarray.index.backends.hnswlib.HnswDocumentIndex) documentation.
    TODO link docs

### Create a Document Index

To create a Document Index, your first need a Document that defines the schema of your index.

```python
from docarray import BaseDoc
from docarray.index import HNSWLibDocumentIndex
from docarray.typing import NdArray


class MyDoc(BaseDoc):
    embedding: NdArray[128]
    text: str


db = HNSWLibDocumentIndex[MyDoc](work_dir='./my_test_db')
```

**Schema definition:**

In this code snippet, `HNSWLibDocumentIndex` takes a schema of the form of `MyDoc`.
The Document Index then _creates column for each field in `MyDoc`_.

The column types in the backend database are determined the type hints of the fields in the Document.
Optionally, you can customize the database types for every field TODO link to this.

Most vector databases need to know the dimensionality of the vectors that will be stored.
Here, that is automatically inferred from the type hint of the `embedding` field: `NdArray[128]` means that
the database will store vectors with 128 dimensions.

!!! note "PyTorch and TensorFlow support"
    Instead of using `NdArray` you can use `TorchTensor` or `TensorFlowTensor` and the Document Index will handle that
    for you. No need to convert your tensors to numpy arrays!

**Database location:**

For `HNSWLibDocumentIndex` you need to specify a `work_dir` where the data will be stored; for other backends you
usually specify a `host` and a `port` instead.

Either way, if the location does not yet contain any data, we start from a blank slate.
If the location already contains data from a previous session, it will be accessible through the Document Index.

### Index data

Now that you have a Document Index, you can add data to it, using the [index()][docarray.index.backends.hnswlib.HnswDocumentIndex.index] method:

```python
import numpy as np
from docarray import DocList

# create some random data
docs = DocList[MyDoc](
    [MyDoc(embedding=np.random.rand(128), text=f'text {i}') for i in range(100)]
)

# index the data
db.index(docs)
```

That call to [index()][docarray.index.backends.hnswlib.HnswDocumentIndex.index] stores all Documents in `docs` into the Document Index,
ready to be retrieved in the next step.

As you can see, `DocList[MyDoc]` and `HNSWLibDocumentIndex[MyDoc]` are both parameterized with `MyDoc`.
This means that they share the same schema, and in general, the schema of a Document Index and the data that you want to store
need to have compatible schemas.

!!! question "When are two schemas compatible?"
    The schema of your Document Index and of your data need to be compatible with each other.
    
    Let's say A is the schema of your Document Index and B is the schema of your data.
    There are a few rules that determine if a schema A is compatible with a schema B.
    If _any_ of the following is true, then A and B are compatible:
    - A and B are the same class
    - A and B have the same field names and field types
    - A and B have the same field names, and, for every field, the type of B is a subclass of the type of A

### Perform vector similarity search

Now that you have indexed your data, you can perform vector similarity search using the [find()][docarray.index.backends.hnswlib.HnswDocumentIndex.find] method.


Provided with a Document of type `MyDoc`, [find()][docarray.index.backends.hnswlib.HnswDocumentIndex.find] can find
similar Documents in the Document Index.

=== "Search by Document"

```python
# create a query Document
query = MyDoc(embedding=np.random.rand(128), text='query')

# find similar Documents
matches, scores = db.find(query, search_field='embedding', limit=5)

print(f'{matches=}')
print(f'{matches.text=}')
print(f'{scores=}')
```

=== "Search by raw vector"

```python
# create a query vector
query = np.random.rand(128)

# find similar Documents
matches, scores = db.find(query, search_field='embedding', limit=5)

print(f'{matches=}')
print(f'{matches.text=}')
print(f'{scores=}')
```

To succesfully peform a vector search, you need to specify a `search_field`. This is the field that serves as the
basis of comparison between your query and the documents in the Document Index.

In this particular example you only have one field (`embedding`) that is a vector, so you can trivially choose that one.
In general, you could have multiple fields of type `NdArray` or `TorchTensor` or `TensorFlowTensor`, and you can choose
which one to use for the search.

The [find()][docarray.index.backends.hnswlib.HnswDocumentIndex.find] method returns a named tuple containing the closest
matching documents and their associated similarity scores.

How these scores are calculated depends on the backend, and can usually be configured TODO link.

**Batched search:**

You can also search for multiple Documents at once, in a batch, using the [find_batched()][docarray.index.backends.hnswlib.HnswDocumentIndex.find_batched] method.

=== "Search by Documents"

```python
# create some query Documents
queries = DocList[MyDoc](
    MyDoc(embedding=np.random.rand(128), text=f'query {i}') for _ in range(3)
)

# find similar Documents
matches, scores = db.find(queries, search_field='embedding', limit=5)

print(f'{matches=}')
print(f'{matches.text=}')
print(f'{scores=}')
```

=== "Search by raw vector"

```python
# create some query vectors
query = np.random.rand(3, 128)

# find similar Documents
matches, scores = db.find(query, search_field='embedding', limit=5)

print(f'{matches=}')
print(f'{matches[0].text=}')
print(f'{scores=}')
```

The [find_batched()][docarray.index.backends.hnswlib.HnswDocumentIndex.find_batched] method returns a named tuple containing
a list of `DocList`s, one for each query, containing the closest matching documents; and the associated similarity scores.

### Perform filter search

You can also perform filter search using the [filter()][docarray.index.backends.hnswlib.HnswDocumentIndex.filter] method.

This method takes in a filter query and returns Documents that fulfill the conditions expressed through that filter:

```python
# create a filter
```

### Delete data

## Document Store
This section show you how to use the `DocArray.store` module. `DocArray.store` module is used to store the `Doc`.

- link to jac
- link to s3
