# Redis Document Index

!!! note "Install dependencies"
    To use [RedisDocumentIndex][docarray.index.backends.redis.RedisDocumentIndex], you need to install extra dependencies with the following command:

    ```console
    pip install "docarray[redis]"
    ```

This is the user guide for the [RedisDocumentIndex][docarray.index.backends.redis.RedisDocumentIndex],
focusing on special features and configurations of Redis.


## Basic usage
This snippet demonstrates the basic usage of [RedisDocumentIndex][docarray.index.backends.redis.RedisDocumentIndex]. It defines a document schema with a title and an embedding, 
creates ten dummy documents with random embeddings, initializes an instance of [RedisDocumentIndex][docarray.index.backends.redis.RedisDocumentIndex] to index these documents, 
and performs a vector similarity search to retrieve the ten most similar documents to a given query vector.

```python
from docarray import BaseDoc, DocList
from docarray.index import RedisDocumentIndex
from docarray.typing import NdArray
import numpy as np

# Define the document schema.
class MyDoc(BaseDoc):
    title: str 
    embedding: NdArray[128]

# Create dummy documents.
docs = DocList[MyDoc](MyDoc(title=f'title #{i}', embedding=np.random.rand(128)) for i in range(10))

# Initialize a new RedisDocumentIndex instance and add the documents to the index.
doc_index = RedisDocumentIndex[MyDoc](host='localhost')
doc_index.index(docs)

# Perform a vector search.
query = np.ones(128)
retrieved_docs = doc_index.find(query, search_field='embedding', limit=10)
```


## Initialize

Before initializing [RedisDocumentIndex][docarray.index.backends.redis.RedisDocumentIndex], 
make sure that you have a Redis service that you can connect to. 

You can create a local Redis service with the following command:

```shell
docker run --name redis-stack-server -p 6379:6379 -d redis/redis-stack-server:7.2.0-RC2
```
Next, you can create [RedisDocumentIndex][docarray.index.backends.redis.RedisDocumentIndex]:
```python
from docarray import BaseDoc
from docarray.index import RedisDocumentIndex
from docarray.typing import NdArray


class MyDoc(BaseDoc):
    embedding: NdArray[128]
    text: str


doc_index = RedisDocumentIndex[MyDoc](host='localhost')
```


### Schema definition
In this code snippet, `RedisDocumentIndex` takes a schema of the form of `MyDoc`.
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

DocArray offers a number of predefined Documents, like [ImageDoc][docarray.documents.ImageDoc] and [TextDoc][docarray.documents.TextDoc].
If you try to use these directly as a schema for a Document Index, you will get unexpected behavior:
Depending on the backend, an exception will be raised, or no vector index for ANN lookup will be built.

The reason for this is that predefined Documents don't hold information about the dimensionality of their `.embedding`
field. But this is crucial information for any vector database to work properly!

You can work around this problem by subclassing the predefined document and adding the dimensionality information:

=== "Using type hint"
    ```python
    from docarray.documents import TextDoc
    from docarray.typing import NdArray
    from docarray.index import RedisDocumentIndex


    class MyDoc(TextDoc):
        embedding: NdArray[128]


    doc_index = RedisDocumentIndex[MyDoc]()
    ```

=== "Using Field()"
    ```python
    from docarray.documents import TextDoc
    from docarray.typing import AnyTensor
    from docarray.index import RedisDocumentIndex
    from pydantic import Field


    class MyDoc(TextDoc):
        embedding: AnyTensor = Field(dim=128)


    doc_index = RedisDocumentIndex[MyDoc]()
    ```

Once you have defined the schema of your Document Index in this way, the data that you index can be either the predefined Document type or your custom Document type.

The [next section](#index) goes into more detail about data indexing, but note that if you have some `TextDoc`s, `ImageDoc`s etc. that you want to index, you _don't_ need to cast them to `MyDoc`:

```python
from docarray import DocList

# data of type TextDoc
data = DocList[TextDoc](
    [
        TextDoc(text='hello world', embedding=np.random.rand(128)),
        TextDoc(text='hello world', embedding=np.random.rand(128)),
        TextDoc(text='hello world', embedding=np.random.rand(128)),
    ]
)

# you can index this into Document Index of type MyDoc
doc_index.index(data)
```

## Index

Now that you have a Document Index, you can add data to it, using the [`index()`][docarray.index.abstract.BaseDocIndex.index] method:

```python
import numpy as np
from docarray import DocList

# create some random data
docs = DocList[MyDoc](
    [MyDoc(embedding=np.random.rand(128), text=f'text {i}') for i in range(100)]
)

# index the data
doc_index.index(docs)
```

That call to [`index()`][docarray.index.backends.redis.RedisDocumentIndex.index] stores all Documents in `docs` in the Document Index,
ready to be retrieved in the next step.

As you can see, `DocList[MyDoc]` and `RedisDocumentIndex[MyDoc]` both have `MyDoc` as a parameter.
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
    query = MyDoc(embedding=np.random.rand(128), text='query')

    # find similar documents
    matches, scores = doc_index.find(query, search_field='embedding', limit=5)

    print(f'{matches=}')
    print(f'{matches.text=}')
    print(f'{scores=}')
    ```

=== "Search by raw vector"

    ```python
    # create a query vector
    query = np.random.rand(128)

    # find similar documents
    matches, scores = doc_index.find(query, search_field='embedding', limit=5)

    print(f'{matches=}')
    print(f'{matches.text=}')
    print(f'{scores=}')
    ```

To peform a vector search, you need to specify a `search_field`. This is the field that serves as the
basis of comparison between your query and the documents in the Document Index.

In this example you only have one field (`embedding`) that is a vector, so you can trivially choose that one.
In general, you could have multiple fields of type `NdArray` or `TorchTensor` or `TensorFlowTensor`, and you can choose
which one to use for the search.

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
    matches, scores = doc_index.find_batched(queries, search_field='embedding', limit=5)

    print(f'{matches=}')
    print(f'{matches[0].text=}')
    print(f'{scores=}')
    ```

=== "Search by raw vectors"

    ```python
    # create some query vectors
    query = np.random.rand(3, 128)

    # find similar documents
    matches, scores = doc_index.find_batched(query, search_field='embedding', limit=5)

    print(f'{matches=}')
    print(f'{matches[0].text=}')
    print(f'{scores=}')
    ```

The [`find_batched()`][docarray.index.abstract.BaseDocIndex.find_batched] method returns a named tuple containing
a list of `DocList`s, one for each query, containing the closest matching documents and their similarity scores.


## Filter

You can filter your documents by using the `filter()` or `filter_batched()` method with a corresponding  filter query. 
The query should follow the [query language of the Redis](https://redis.io/docs/interact/search-and-query/query/).

In the following example let's filter for all the books that are cheaper than 29 dollars:

```python
from docarray import BaseDoc, DocList


class Book(BaseDoc):
    title: str
    price: int


books = DocList[Book]([Book(title=f'title {i}', price=i * 10) for i in range(10)])
book_index = RedisDocumentIndex[Book]()
book_index.index(books)

# filter for books that are cheaper than 29 dollars
query = '@price:[-inf 29]'
cheap_books = book_index.filter(filter_query=query)

assert len(cheap_books) == 3
for doc in cheap_books:
    doc.summary()
```

## Text search

In addition to vector similarity search, the Document Index interface offers methods for text search:
[`text_search()`][docarray.index.abstract.BaseDocIndex.text_search],
as well as the batched version [`text_search_batched()`][docarray.index.abstract.BaseDocIndex.text_search_batched].

You can use text search directly on the field of type `str`:

```python
class NewsDoc(BaseDoc):
    text: str


doc_index = RedisDocumentIndex[NewsDoc]()
index_docs = [
    NewsDoc(id='0', text='this is a news for sport'),
    NewsDoc(id='1', text='this is a news for finance'),
    NewsDoc(id='2', text='this is another news for sport'),
]
doc_index.index(index_docs)
query = 'finance'

# search with text
docs, scores = doc_index.text_search(query, search_field='text')
```

## Hybrid search

Document Index supports atomic operations for vector similarity search, text search and filter search.

To combine these operations into a single, hybrid search query, you can use the query builder that is accessible
through [`build_query()`][docarray.index.abstract.BaseDocIndex.build_query]:

```python
# Define the document schema.
class SimpleSchema(BaseDoc):
    price: int
    embedding: NdArray[128]

# Create dummy documents.
docs = DocList[SimpleSchema](SimpleSchema(price=i, embedding=np.random.rand(128)) for i in range(10))

doc_index = RedisDocumentIndex[SimpleSchema](host='localhost')
doc_index.index(docs)

query = (
    doc_index.build_query()  # get empty query object
    .find(query=np.random.rand(128), search_field='embedding')  # add vector similarity search
    .filter(filter_query='@price:[-inf 3]')  # add filter search
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
    MyDoc(embedding=np.random.rand(128), text=f'query {i}') for i in range(3)
)

# remember the Document ids and index the data
ids = data.id
db.index(data)

# access the Documents by id
doc = db[ids[0]]  # get by single id
docs = db[ids]  # get by list of ids
```


## Delete documents

In the same way you can access Documents by `id`, you can also delete them:

```python
# prepare some data
data = DocList[MyDoc](
    MyDoc(embedding=np.random.rand(128), text=f'query {i}') for i in range(3)
)

# remember the Document ids and index the data
ids = data.id
db.index(data)

# access the Documents by id
del db[ids[0]]  # del by single id
del db[ids[1:]]  # del by list of ids
```

## Update documents
In order to update a Document inside the index, you only need to re-index it with the updated attributes.

First, let's create a schema for our Document Index:
```python
import numpy as np
from docarray import BaseDoc, DocList
from docarray.typing import NdArray
from docarray.index import RedisDocumentIndex
class MyDoc(BaseDoc):
    text: str
    embedding: NdArray[128]
```

Now, we can instantiate our Index and add some data:
```python
docs = DocList[MyDoc](
    [MyDoc(embedding=np.random.rand(128), text=f'I am the first version of Document {i}') for i in range(100)]
)
index = RedisDocumentIndex[MyDoc]()
index.index(docs)
assert index.num_docs() == 100
```

Let's retrieve our data and check its content:
```python
res = index.find(query=docs[0], search_field='embedding', limit=100)
assert len(res.documents) == 100
for doc in res.documents:
    assert 'I am the first version' in doc.text
```

Then, let's update all of the text of these documents and re-index them:
```python
for i, doc in enumerate(docs):
    doc.text = f'I am the second version of Document {i}'

index.index(docs)
assert index.num_docs() == 100
```

When we retrieve them again we can see that their text attribute has been updated accordingly:
```python
res = index.find(query=docs[0], search_field='embedding', limit=100)
assert len(res.documents) == 100
for doc in res.documents:
    assert 'I am the second version' in doc.text
```


## Configuration

This section lays out the configurations and options that are specific to [RedisDocumentIndex][docarray.index.backends.redis.RedisDocumentIndex].

### DBConfig

The following configs can be set in `DBConfig`:

| Name                    | Description                                        | Default                                                                             |
|-------------------------|----------------------------------------------------|-------------------------------------------------------------------------------------|
| `host`                  | The host address for the Redis server.             | `localhost`                                                                         |
| `port`                  | The port number for the Redis server               | 6379                                                                                |
| `index_name`            | The name of the index in the Redis database        | `None`. Data will be stored in an index named after the Document type used as schema. |
| `username`              | The username for the Redis server                  | `None`                                                                                |
| `password`              | The password for the Redis server                  | `None`                                                                                |
| `text_scorer`           | The method for [scoring text](https://redis.io/docs/interact/search-and-query/advanced-concepts/scoring/) during text search | `BM25`                                                                              |
| `default_column_config` | The default configurations for every column type.  | dict                                                                                |

You can pass any of the above as keyword arguments to the `__init__()` method or pass an entire configuration object.


### Field-wise configuration


`default_column_config` is the default configurations for every column type. Since there are many column types in Redis, you can also consider changing the column config when defining the schema.

```python
class SimpleDoc(BaseDoc):
    tensor: NdArray[128] = Field(algorithm='FLAT', distance='COSINE')


doc_index = RedisDocumentIndex[SimpleDoc]()
```


### RuntimeConfig

The `RuntimeConfig` dataclass of `RedisDocumentIndex` consists of `batch_size` index/get/del operations. 
You can change `batch_size` in the following way:

```python
doc_index = RedisDocumentIndex[SimpleDoc]()
doc_index.configure(RedisDocumentIndex.RuntimeConfig(batch_size=128))
```

You can pass the above as keyword arguments to the `configure()` method or pass an entire configuration object.



## Nested data and subindex search

The examples provided primarily operate on a basic schema where each field corresponds to a straightforward type such as `str` or `NdArray`. 
However, it is also feasible to represent and store nested documents in a Document Index, including scenarios where a document 
contains a `DocList` of other documents. 

Go to the [Nested Data](nested_data.md) section to learn more.