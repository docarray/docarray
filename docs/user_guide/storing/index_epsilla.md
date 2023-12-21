# Epsilla Document Index

!!! note "Install dependencies"
    To use [EpsillaDocumentIndex][docarray.index.backends.epsilla.EpsillaDocumentIndex], you need to install extra dependencies with the following command:

    ```console
        pip install "docarray[epsilla]"
        pip install --upgrade pyepsilla
    ```

## Basic usage

This snippet demonstrates the basic usage of
[EpsillaDocumentIndex][docarray.index.backends.epsilla.EpsillaDocumentIndex]:

1. Define a document schema with two fields: title and embedding
2. Create ten dummy documents with random embeddings
3. Set the db config and initialize the index
4. Add dummy documents to the index
5. Finally, perform a vector similarity search to retrieve the ten most similar documents to a given query vector

```python
from docarray import BaseDoc, DocList
from docarray.index.backends.epsilla import EpsillaDocumentIndex
from docarray.typing import NdArray
from pydantic import Field
import numpy as np


# Define the document schema.
class MyDoc(BaseDoc):
    title: str
    embedding: NdArray[128] = Field(is_embedding=True)


# Create dummy documents.
docs = DocList[MyDoc](
    MyDoc(title=f'title #{i}', embedding=np.random.rand(128)) for i in range(10)
)

# db_config, see the initialize section below
db_config = EpsillaDocumentIndex.DBConfig(
    is_self_hosted=True,
    protocol="http",
    host="localhost",
    port=8888,
    db_path="/epsilla",
    db_name="test",
)

# Initialize a new EpsillaDocumentIndex instance
doc_index = EpsillaDocumentIndex[MyDoc](db_config=db_config)

# Add the documents to the index.
doc_index.index(docs)

# Perform a vector search.
query = MyDoc(title="test", embedding=np.ones(128))
retrieved_docs = doc_index.find(query, limit=10, search_field="embedding")
print(f'{retrieved_docs=}')
retrieved_docs[0].summary()
```

The following sections will cover details of the individual steps.

## Initialize

### Start and connect to Epsilla

To use [EpsillaDocumentIndex][docarray.index.backends.epsilla.EpsillaDocumentIndex], DocArray needs to hook into a
running Epsilla service.
There are multiple ways to start a Epsilla instance, depending on your use case.

**Options - Overview**

| Instance type      | General use case           | Configurability | Notes                          |
| ------------------ | -------------------------- | --------------- | ------------------------------ |
| **Epsilla Cloud ** | Development and production | Limited         | **Recommended for most users** |
| **Docker**         | Self hosted                | Full            |                                |

**Connect via Epsilla Cloud**

Check out [Epsilla's documentation](https://epsilla-inc.gitbook.io/epsilladb/quick-start/epsilla-cloud) to create an
instance, and for information on obtaining your credentials.

**Connect via Docker (self-managed)**

```bash
docker pull epsilla/vectordb
```

Start the docker as the backend service

```bash
docker run --pull=always -d -p 8888:8888 epsilla/vectordb
```

### Connecting to Epsilla

**Cloud instance**

Check out [Epsilla's documentation](https://epsilla-inc.gitbook.io/epsilladb/quick-start/epsilla-cloud) for credentials.

```python
from docarray.index.backends.epsilla import EpsillaDocumentIndex

db = EpsillaDocumentIndex.DBConfig(
    is_self_hosted=False,
    cloud_project_id="your-project-id",
    cloud_db_id="your-database-id",
    api_key="your-epsilla-api-key",
)
```

**Self hosted**

```python
from docarray.index.backends.epsilla import EpsillaDocumentIndex

db = EpsillaDocumentIndex.DBConfig(
    is_self_hosted=True,
    protocol=None,
    host="localhost",
    port=8888,
    db_path=None,
    db_name=None,
)
```

### Create an instance

Let's connect to a local Epsilla container and instantiate a `EpsillaDocumentIndex` instance for a given schema:

```python
from docarray import BaseDoc
from docarray.index.backends.epsilla import EpsillaDocumentIndex
from docarray.typing import NdArray
from pydantic import Field


# Define the document schema.
class MyDoc(BaseDoc):
    title: str
    embedding: NdArray[128] = Field(is_embedding=True)


# Set the database configuration.
db_config = EpsillaDocumentIndex.DBConfig(
    is_self_hosted=True,
    protocol="http",
    host="localhost",
    port=8888,
    db_path="/epsilla",
    db_name="test",
)

# Initialize a new EpsillaDocumentIndex instance
doc_index = EpsillaDocumentIndex[MyDoc](db_config=db_config)
```

### Schema definition

In this code snippet, `EpsillaDocumentIndex` takes a schema of the form of `MyDoc`.
The Document Index then _creates a column for each field in `MyDoc`_.

The column types in the backend database are determined by the type hints of the document's fields.
Optionally, you can [customize the database types for every field](#configuration).

Most vector databases need to know the dimensionality of the vectors that will be stored.
Here, that is automatically inferred from the type hint of the `embedding` field: `NdArray[128]` means that
the database will store vectors with 128 dimensions.

!!! note "PyTorch and TensorFlow support"
    Instead of using `NdArray` you can use `TorchTensor` or `TensorFlowTensor` and the Document Index will handle that
    for you. This is supported for all Document Index backends. No need to convert your tensors to NumPy arrays manually!

### Using a predefined document as schema

DocArray offers a number of predefined documents, like [ImageDoc][docarray.documents.ImageDoc]
and [TextDoc][docarray.documents.TextDoc].
If you try to use these directly as a schema for a Document Index, you will get unexpected behavior:
Depending on the backend, an exception will be raised, or no vector index for ANN lookup will be built.

The reason for this is that predefined documents don't hold information about the dimensionality of their `.embedding`
field. But this is crucial information for any vector database to work properly!

You can work around this problem by subclassing the predefined document and adding the dimensionality information:

=== "Using type hint"

```python
from docarray.documents import TextDoc
from docarray.typing import NdArray
from docarray.index import EpsillaDocumentIndex
from pydantic import Field


class MyDoc(TextDoc):
    embedding: NdArray[128] = Field(is_embedding=True)


doc_index = EpsillaDocumentIndex[MyDoc]()
```

=== "Using Field()"

```python
from docarray.documents import TextDoc
from docarray.typing import AnyTensor
from docarray.index import EpsillaDocumentIndex
from pydantic import Field


class MyDoc(TextDoc):
    embedding: AnyTensor = Field(dim=128, is_embedding=True)


doc_index = EpsillaDocumentIndex[MyDoc]()
```

Once you have defined the schema of your Document Index in this way, the
data that you index can be either the predefined Document type or your custom Document type.

The [next section]( # index) goes into more detail about data indexing, but note that if you have some `TextDoc`
, `ImageDoc` etc. that you want to index, you _don't_ need to cast them to `MyDoc`:

```python
from docarray import DocList

data = DocList[MyDoc](
    [
        MyDoc(title='hello world', embedding=np.random.rand(128)),
        MyDoc(title='hello world', embedding=np.random.rand(128)),
        MyDoc(title='hello world', embedding=np.random.rand(128)),
    ]
)

# you can index this into Document Index of type MyDoc
doc_index.index(data)
```

## Index

Now that you have a Document Index, you can add data to it, using
the [`index()`][docarray.index.abstract.BaseDocIndex.index] method:

```python
from docarray import BaseDoc, DocList
from docarray.index.backends.epsilla import EpsillaDocumentIndex
from docarray.typing import NdArray
from pydantic import Field
import numpy as np


class MyDoc(BaseDoc):
    title: str
    embedding: NdArray[128] = Field(is_embedding=True)


# Create dummy documents.
docs = DocList[MyDoc](
    MyDoc(title=f'title #{i}', embedding=np.random.rand(128)) for i in range(10)
)

db_config = "..."  # see the initialize section above

doc_index = EpsillaDocumentIndex[MyDoc](db_config=db_config, index_name='mydoc_index')

# add the data
doc_index.index(docs)
```

That call to [`index()`][docarray.index.backends.epsilla.EpsillaDocumentIndex.index] stores all Documents in `docs` in
the Document Index,
ready to be retrieved in the next step.

As you can see, `DocList[Document]` and `EpsillaDocumentIndex[Document]` both have `Document` as a parameter.
This means that they share the same schema, and in general, both the Document Index and the data that you want to store
need to have compatible schemas.

!!! question "When are two schemas compatible?"
    The schemas of your Document Index and data need to be compatible with each other.

    Let's say A is the schema of your Document Index and B is the schema of your data.
    There are a few rules that determine if schema A is compatible with schema B.
    If _any_ of the following are true, then A and B are compatible:

    - A and B are the same class
    - A and B have the same field names and field types
    - A and B have the same field names, and, for every field, the type of B is a subclass of the type of A

    In particular, this means that you can easily [index predefined documents](#using-a-predefined-document-as-schema) into a Document Index.

## Vector search

Now that you have indexed your data, you can perform vector similarity search using
the [`find()`][docarray.index.abstract.BaseDocIndex.find] method.

You can perform a similarity search and find relevant documents by passing `MyDoc` or a raw vector to
the [`find()`][docarray.index.abstract.BaseDocIndex.find] method:

=== "Search by Document"

    ```python
    # create a query document
    query = Document(
        text="Hello world",
        embedding=np.array([1, 2]),
        file=np.random.rand(100),
    )

    # find similar documents
    matches, scores = doc_index.find(query, limit=5)

    print(f"{matches=}")
    print(f"{matches.text=}")
    print(f"{scores=}")
    ```

=== "Search by raw vector"

    ```python
    # create a query vector
    query = np.random.rand(2)

    # find similar documents
    matches, scores = store.find(query, limit=5)

    print(f'{matches=}')
    print(f'{matches.text=}')
    print(f'{scores=}')
    ```

The [`find()`][docarray.index.abstract.BaseDocIndex.find] method returns a named tuple containing the closest
matching documents and their associated similarity scores.

When searching on the subindex level, you can use
the [`find_subindex()`][docarray.index.abstract.BaseDocIndex.find_subindex] method, which returns a named tuple
containing the subindex documents, similarity scores and their associated root documents.

How these scores are calculated depends on the backend, and can usually be [configured](#configuration).

### Batched search

You can also search for multiple documents at once, in a batch, using
the [`find_batched()`][docarray.index.abstract.BaseDocIndex.find_batched] method.

=== "Search by documents"

    ```python
    # create some query documents
    queries = DocList[MyDoc](
        Document(
            text=f"Hello world {i}",
            embedding=np.array([i, i + 1]),
            file=np.random.rand(100),
        )
        for i in range(3)
    )

    # find similar documents
    matches, scores = doc_index.find_batched(queries, limit=5)

    print(f"{matches=}")
    print(f"{matches[0].text=}")
    print(f"{scores=}")
    ```

=== "Search by raw vectors"

    ```python
    # create some query vectors
    query = np.random.rand(3, 2)

    # find similar documents
    matches, scores = doc_index.find_batched(query, limit=5)

    print(f'{matches=}')
    print(f'{matches[0].text=}')
    print(f'{scores=}')
    ```

The [`find_batched()`][docarray.index.abstract.BaseDocIndex.find_batched] method returns a named tuple containing
a list of `DocList`s, one for each query, containing the closest matching documents and their similarity scores.

## Filter

To perform filtering, follow the below syntax.

This will perform a filtering on the field `title`:

```python
docs = doc_index.filter("title = 'test'", limit=5)
```

You can filter your documents by using the `filter()` or `filter_batched()` method with a corresponding filter query.
The query should follow the [filters supported by Epsilla](https://epsilla-inc.gitbook.io/epsilladb/vector-database/search-the-top-k-semantically-similar-records#filter-expression).

In the following example let's filter for all the books that are cheaper than 29 dollars:

```python
from docarray import BaseDoc, DocList
from docarray.index.backends.epsilla import EpsillaDocumentIndex
from docarray.typing import NdArray
from pydantic import Field
import numpy as np


class Book(BaseDoc):
    price: int
    embedding: NdArray[10] = Field(is_embedding=True)


books = DocList[Book](
    [Book(price=i * 10, embedding=np.random.rand(10)) for i in range(10)]
)
db_config = "..."  # see the initialize section above
book_index = EpsillaDocumentIndex[Book](db_config=db_config, index_name='tmp_index')
book_index.index(books)

# filter for books that are cheaper than 29 dollars
query = "price < 29"
cheap_books = book_index.filter(filter_query=query)
print(f"{cheap_books=}")
cheap_books[0].summary()
```

## Text search

!!! warning
    The [EpsillaDocumentIndex][docarray.index.backends.epsilla.EpsillaDocumentIndex] implementation does not support text
    search.

## Hybrid search

Document Index supports atomic operations for vector similarity search, text search and filter search.

To combine these operations into a single, hybrid search query, you can use the query builder that is accessible
through [`build_query()`][docarray.index.abstract.BaseDocIndex.build_query]:

```python
# Define the document schema.
class SimpleSchema(BaseDoc):
    year: int
    price: int
    embedding: NdArray[128]


# Create dummy documents.
docs = DocList[SimpleSchema](
    SimpleSchema(year=2000 - i, price=i, embedding=np.random.rand(128))
    for i in range(10)
)

doc_index = EpsillaDocumentIndex[SimpleSchema]()
doc_index.index(docs)

query = (
    doc_index.build_query()  # get empty query object
    .filter(filter_query="year>1994")  # pre-filtering
    .find(
        query=np.random.rand(128), search_field='embedding'
    )  # add vector similarity search
    .filter(filter_query="price<3")  # post-filtering
    .build()
)
# execute the combined query and return the results
results = doc_index.execute_query(query)
print(f'{results=}')
```

In the example above you can see how to form a hybrid query that combines vector similarity search and filtered search
to obtain a combined set of results.

The kinds of atomic queries that can be combined in this way depends on the backend.
Some backends can combine text search and vector search, while others can perform filters and vectors search, etc.

## Access documents

To retrieve a document from a Document Index you don't necessarily need to perform a fancy search.

You can also access data by the `id` that was assigned to each document:

```python
# prepare some data
data = DocList[MyDoc](
    MyDoc(embedding=np.random.rand(128), title=f'query {i}') for i in range(3)
)

# remember the Document ids and index the data
ids = data.id
doc_index.index(data)

# access the documents by id
doc = doc_index[ids[0]]  # get by single id
docs = doc_index[ids]  # get by list of ids
```

## Delete documents

In the same way you can access documents by `id`, you can also delete them:

```python
# prepare some data
data = DocList[MyDoc](
    MyDoc(embedding=np.random.rand(128), title=f'query {i}') for i in range(3)
)

# remember the Document ids and index the data
ids = data.id
doc_index.index(data)

# access the documents by id
del doc_index[ids[0]]  # del by single id
del doc_index[ids[1:]]  # del by list of ids
```

## Count documents

!!! warning
    Unlike other index backends, Epsilla does not provide a count API. When using it with docarray, calling the `num_docs` method will raise errors.

    ```python
    # will raise errors
    doc_index.num_docs()
    ```

If you need to count how many documents there are in the index, you can try to use the filter method.

```python
# use a larger limit as needed
doc_index.filter(filter_query="", limit=100)
```

## Configuration

### DBConfig

The following configs can be set in `DBConfig`:

| Name               | Description                                   | Default |
| ------------------ | --------------------------------------------- | ------- |
| `is_self_hosted`   | If using Epsilla cloud or running self hosted | `false` |
| `cloud_project_id` | If using Epsilla cloud; found in the console  | `None`  |
| `cloud_db_id`      | If using Epsilla cloud; found in the console  | `None`  |
| `api_key`          | If using Epsilla cloud; found in the console  | `None`  |
| `host`             | Address or 'localhost'                        | `None`  |
| `port`             | The port number for the Epsilla server        | 8888    |
| `protocol`         | Protocol to connect, e.g. 'http'              | `None`  |
| `db_path`          | Path to the database on disk                  | `None`  |
| `db_name`          | Name of the database                          | `None`  |

You can pass any of the above as keyword arguments to the `__init__()` method or pass an entire configuration object.

## Nested data and subindex search

The examples provided primarily operate on a basic schema where each field corresponds to a straightforward type such  
as `str` or `NdArray`.
However, it is also feasible to represent and store nested documents in a Document Index, including scenarios where a
document contains a `DocList` of other documents.

Go to the [Nested Data](nested_data.md) section to learn more.
