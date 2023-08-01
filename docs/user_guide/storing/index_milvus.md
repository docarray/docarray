# Milvus Document Index

!!! note "Install dependencies"
    To use [MilvusDocumentIndex][docarray.index.backends.milvus.MilvusDocumentIndex], you need to install extra dependencies with the following command:

    ```console
    pip install "docarray[milvus]"
    ```

This is the user guide for the [MilvusDocumentIndex][docarray.index.backends.milvus.MilvusDocumentIndex],
focusing on special features and configurations of Milvus.


## Basic usage
This snippet demonstrates the basic usage of [MilvusDocumentIndex][docarray.index.backends.milvus.MilvusDocumentIndex]. It defines a document schema with a title and an embedding, 
creates ten dummy documents with random embeddings, initializes an instance of [MilvusDocumentIndex][docarray.index.backends.milvus.MilvusDocumentIndex] to index these documents, 
and performs a vector similarity search to retrieve ten most similar documents to a given query vector.

!!! note "Single Search Field Requirement"
    In order to utilize vector search, it's necessary to define 'is_embedding' for one field only. 
    This is due to Milvus' configuration, which permits a single vector for each data object.

```python
from docarray import BaseDoc, DocList
from docarray.index import MilvusDocumentIndex
from docarray.typing import NdArray
from pydantic import Field
import numpy as np

# Define the document schema.
class MyDoc(BaseDoc):
    title: str 
    embedding: NdArray[128] = Field(is_embedding=True)

# Create dummy documents.
docs = DocList[MyDoc](MyDoc(title=f'title #{i}', embedding=np.random.rand(128)) for i in range(10))

# Initialize a new MilvusDocumentIndex instance and add the documents to the index.
doc_index = MilvusDocumentIndex[MyDoc](index_name='tmp_index_1')
doc_index.index(docs)

# Perform a vector search.
query = np.ones(128)
retrieved_docs = doc_index.find(query, limit=10)
```


## Initialize

First of all, you need to install and run Milvus. Download `docker-compose.yml` with the following command:

```shell
wget https://github.com/milvus-io/milvus/releases/download/v2.2.11/milvus-standalone-docker-compose.yml -O docker-compose.yml
```

And start Milvus by running:
```shell
sudo docker-compose up -d
```

Learn more on [Milvus documentation](https://milvus.io/docs/install_standalone-docker.md).

Next, you can create a [MilvusDocumentIndex][docarray.index.backends.milvus.MilvusDocumentIndex] instance using:

```python
from docarray import BaseDoc
from docarray.index import MilvusDocumentIndex
from docarray.typing import NdArray
from pydantic import Field


class MyDoc(BaseDoc):
    embedding: NdArray[128] = Field(is_embedding=True)
    text: str


doc_index = MilvusDocumentIndex[MyDoc](index_name='tmp_index_2')
```

### Schema definition
In this code snippet, `MilvusDocumentIndex` takes a schema of the form of `MyDoc`.
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

DocArray offers a number of predefined documents, like [ImageDoc][docarray.documents.ImageDoc] and [TextDoc][docarray.documents.TextDoc].
If you try to use these directly as a schema for a Document Index, you will get unexpected behavior:
Depending on the backend, an exception will be raised, or no vector index for ANN lookup will be built.

The reason for this is that predefined Documents don't hold information about the dimensionality of their `.embedding`
field. But this is crucial information for any vector database to work properly!

You can work around this problem by subclassing the predefined document and adding the dimensionality information:

=== "Using type hint"
    ```python
    from docarray.documents import TextDoc
    from docarray.typing import NdArray
    from docarray.index import MilvusDocumentIndex
    from pydantic import Field


    class MyDoc(TextDoc):
        embedding: NdArray[128] = Field(is_embedding=True)


    doc_index = MilvusDocumentIndex[MyDoc](index_name='tmp_index_3')
    ```

=== "Using Field()"
    ```python
    from docarray.documents import TextDoc
    from docarray.typing import AnyTensor
    from docarray.index import MilvusDocumentIndex
    from pydantic import Field


    class MyDoc(TextDoc):
        embedding: AnyTensor = Field(dim=128, is_embedding=True)


    doc_index = MilvusDocumentIndex[MyDoc](index_name='tmp_index_4')
    ```


## Index

Now that you have a Document Index, you can add data to it, using the [`index()`][docarray.index.abstract.BaseDocIndex.index] method:

```python
import numpy as np
from docarray import DocList

class MyDoc(BaseDoc):
    title: str 
    embedding: NdArray[128] = Field(is_embedding=True)

doc_index = MilvusDocumentIndex[MyDoc](index_name='tmp_index_5')

# create some random data
docs = DocList[MyDoc](
    [MyDoc(embedding=np.random.rand(128), title=f'text {i}') for i in range(100)]
)

# index the data
doc_index.index(docs)
```

That call to [`index()`][docarray.index.backends.milvus.MilvusDocumentIndex.index] stores all Documents in `docs` in the Document Index,
ready to be retrieved in the next step.

As you can see, `DocList[MyDoc]` and `MilvusDocumentIndex[MyDoc]` both have `MyDoc` as a parameter.
This means that they share the same schema, and in general, both the Document Index and the data that you want to store need to have compatible schemas.

!!! question "When are two schemas compatible?"
    The schemas of your Document Index and data need to be compatible with each other.
    
    Let's say A is the schema of your Document Index and B is the schema of your data.
    There are a few rules that determine if schema A is compatible with schema B.
    If _any_ of the following are true, then A and B are compatible:

    - A and B are the same class
    - A and B have the same field names and field types
    - A and B have the same field names, and, for every field, the type of B is a subclass of the type of A

    In particular, this means that you can easily [index predefined Documents](#using-a-predefined-document-as-schema) into a Document Index.


## Vector search

Now that you have indexed your data, you can perform vector similarity search using the [`find()`][docarray.index.abstract.BaseDocIndex.find] method.

You can perform a similarity search and find relevant documents by passing `MyDoc` or a raw vector to 
the [`find()`][docarray.index.abstract.BaseDocIndex.find] method:

=== "Search by Document"

    ```python
    # create a query document
    query = MyDoc(embedding=np.random.rand(128), title='query')

    # find similar documents
    matches, scores = doc_index.find(query, limit=5)

    print(f'{matches=}')
    print(f'{matches.title=}')
    print(f'{scores=}')
    ```

=== "Search by raw vector"

    ```python
    # create a query vector
    query = np.random.rand(128)

    # find similar documents
    matches, scores = doc_index.find(query, limit=5)

    print(f'{matches=}')
    print(f'{matches.title=}')
    print(f'{scores=}')
    ```

The [`find()`][docarray.index.abstract.BaseDocIndex.find] method returns a named tuple containing the closest
matching documents and their associated similarity scores.

When searching on the subindex level, you can use the [`find_subindex()`][docarray.index.abstract.BaseDocIndex.find_subindex] method, which returns a named tuple containing the subindex documents, similarity scores and their associated root documents.

How these scores are calculated depends on the backend, and can usually be [configured](#configuration).

### Batched search

You can also search for multiple documents at once, in a batch, using the [`find_batched()`][docarray.index.abstract.BaseDocIndex.find_batched] method.

=== "Search by documents"

    ```python
    # create some query documents
    queries = DocList[MyDoc](
        MyDoc(embedding=np.random.rand(128), text=f'query {i}') for i in range(3)
    )

    # find similar documents
    matches, scores = doc_index.find_batched(queries, limit=5)

    print(f'{matches=}')
    print(f'{matches[0].text=}')
    print(f'{scores=}')
    ```

=== "Search by raw vectors"

    ```python
    # create some query vectors
    query = np.random.rand(3, 128)

    # find similar documents
    matches, scores = doc_index.find_batched(query, limit=5)

    print(f'{matches=}')
    print(f'{matches[0].text=}')
    print(f'{scores=}')
    ```

The [`find_batched()`][docarray.index.abstract.BaseDocIndex.find_batched] method returns a named tuple containing
a list of `DocList`s, one for each query, containing the closest matching documents and their similarity scores.


## Filter

You can filter your documents by using the `filter()` or `filter_batched()` method with a corresponding  filter query. 
The query should follow the [query language of the Milvus](https://milvus.io/docs/boolean.md).

In the following example let's filter for all the books that are cheaper than 29 dollars:

```python
from docarray import BaseDoc, DocList


class Book(BaseDoc):
    price: int
    embedding: NdArray[10] = Field(is_embedding=True)


books = DocList[Book]([Book(price=i * 10, embedding=np.random.rand(10)) for i in range(10)])
book_index = MilvusDocumentIndex[Book](index_name='tmp_index_6')
book_index.index(books)

# filter for books that are cheaper than 29 dollars
query = 'price < 29'
cheap_books = book_index.filter(filter_query=query)

assert len(cheap_books) == 3
for doc in cheap_books:
    doc.summary()
```

## Text search

!!! note
    The [MilvusDocumentIndex][docarray.index.backends.milvus.MilvusDocumentIndex] implementation does not support text search.

    To see how to perform text search, you can check out other backends that offer support.

In addition to vector similarity search, the Document Index interface offers methods for text search:
[`text_search()`][docarray.index.abstract.BaseDocIndex.text_search],
as well as the batched version [`text_search_batched()`][docarray.index.abstract.BaseDocIndex.text_search_batched].



## Hybrid search

Document Index supports atomic operations for vector similarity search, text search and filter search.

To combine these operations into a single, hybrid search query, you can use the query builder that is accessible
through [`build_query()`][docarray.index.abstract.BaseDocIndex.build_query]:

```python
# Define the document schema.
class SimpleSchema(BaseDoc):
    price: int
    embedding: NdArray[128] = Field(is_embedding=True)

# Create dummy documents.
docs = DocList[SimpleSchema](SimpleSchema(price=i, embedding=np.random.rand(128)) for i in range(10))

doc_index = MilvusDocumentIndex[SimpleSchema](index_name='tmp_index_7')
doc_index.index(docs)

query = (
    doc_index.build_query()  # get empty query object
    .find(query=np.random.rand(128))  # add vector similarity search
    .filter(filter_query='price < 3')  # add filter search
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
data = DocList[SimpleSchema](
    SimpleSchema(embedding=np.random.rand(128), price=i) for i in range(3)
)

# remember the Document ids and index the data
ids = data.id
doc_index.index(data)

# access the Documents by id
doc = doc_index[ids[0]]  # get by single id
docs = doc_index[ids]  # get by list of ids
```


## Delete documents

In the same way you can access Documents by `id`, you can also delete them:

```python
# prepare some data
data = DocList[SimpleSchema](
    SimpleSchema(embedding=np.random.rand(128), price=i) for i in range(3)
)

# remember the Document ids and index the data
ids = data.id
doc_index.index(data)

# access the Documents by id
del doc_index[ids[0]]  # del by single id
del doc_index[ids[1:]]  # del by list of ids
```


## Configuration

This section lays out the configurations and options that are specific to [MilvusDocumentIndex][docarray.index.backends.milvus.MilvusDocumentIndex].

### DBConfig

The following configs can be set in `DBConfig`:

| Name                    | Description                                                                                                                  | Default                                                                        |
|-------------------------|------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| `host`                  | The host address for the Milvus server.                                                                                      | `localhost`                                                                    |
| `port`                  | The port number for the Milvus server                                                                                         | 19530                                                                          |
| `index_name`            | The name of the index in the Milvus database                                                                                  | `None`. Data will be stored in an index named after the Document type used as schema. |
| `user`              | The username for the Milvus server                                                                                            | `None`                                                                           |
| `password`              | The password for the Milvus server                                                                                            | `None`                                                                           |
| `token`              | Token for secure connection                                                                                            | ''                                                                             |
| `collection_description`              | Description of the collection in the database                                                                                            | ''                                                                             |
| `default_column_config` | The default configurations for every column type.                                                                            | dict                                                                           |

You can pass any of the above as keyword arguments to the `__init__()` method or pass an entire configuration object.


### Field-wise configuration


`default_column_config` is the default configurations for every column type. Since there are many column types in Milvus, you can also consider changing the column config when defining the schema.

```python
class SimpleDoc(BaseDoc):
    tensor: NdArray[128] = Field(is_embedding=True, index_type='IVF_FLAT', metric_type='L2')


doc_index = MilvusDocumentIndex[SimpleDoc](index_name='tmp_index_10')
```


### RuntimeConfig

The `RuntimeConfig` dataclass of `MilvusDocumentIndex` consists of `batch_size` index/get/del operations. 
You can change `batch_size` in the following way:

```python
doc_index = MilvusDocumentIndex[SimpleDoc]()
doc_index.configure(MilvusDocumentIndex.RuntimeConfig(batch_size=128))
```

You can pass the above as keyword arguments to the `configure()` method or pass an entire configuration object.


## Nested data and subindex search

The examples provided primarily operate on a basic schema where each field corresponds to a straightforward type such as `str` or `NdArray`. 
However, it is also feasible to represent and store nested documents in a Document Index, including scenarios where a document 
contains a `DocList` of other documents. 

Go to the [Nested Data](nested_data.md) section to learn more.