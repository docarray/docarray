# Qdrant Document Index

!!! note "Install dependencies"
    To use [QdrantDocumentIndex][docarray.index.backends.qdrant.QdrantDocumentIndex], you need to install extra dependencies with the following command:

    ```console
    pip install "docarray[qdrant]"
    ```

The following is a starter script for using the [QdrantDocumentIndex][docarray.index.backends.qdrant.QdrantDocumentIndex],
based on the [Qdrant](https://qdrant.tech/) vector search engine.


## Basic usage
This snippet demonstrates the basic usage of [QdrantDocumentIndex][docarray.index.backends.qdrant.QdrantDocumentIndex]. It defines a document schema with a title and an embedding, 
creates ten dummy documents with random embeddings, initializes an instance of [QdrantDocumentIndex][docarray.index.backends.qdrant.QdrantDocumentIndex] to index these documents, 
and performs a vector similarity search to retrieve the ten most similar documents to a given query vector.

```python
from docarray import BaseDoc, DocList
from docarray.index import QdrantDocumentIndex
from docarray.typing import NdArray
import numpy as np


# Define the document schema.
class MyDoc(BaseDoc):
    title: str
    embedding: NdArray[128]


# Create dummy documents.
docs = DocList[MyDoc](
    MyDoc(title=f'title #{i}', embedding=np.random.rand(128)) for i in range(10)
)

# Initialize a new QdrantDocumentIndex instance and add the documents to the index.
doc_index = QdrantDocumentIndex[MyDoc](host='localhost')
doc_index.index(docs)

# Perform a vector search.
query = np.ones(128)
retrieved_docs, scores = doc_index.find(query, search_field='embedding', limit=10)
```

## Initialize

You can initialize [QdrantDocumentIndex][docarray.index.backends.qdrant.QdrantDocumentIndex] in three different ways:


**Connecting to a local Qdrant instance running as a Docker container**

You can use docker compose to create a local Qdrant service with the following `docker-compose.yml`.

```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:v1.1.2
    ports:
      - "6333:6333"
      - "6334:6334"
    ulimits: # Only required for tests, as there are a lot of collections created
      nofile:
        soft: 65535
        hard: 65535
```

Run the following command in the folder of the above `docker-compose.yml` to start the service:

```bash
docker compose up
```

Next, you can create a [QdrantDocumentIndex][docarray.index.backends.qdrant.QdrantDocumentIndex] instance using:

```python
qdrant_config = QdrantDocumentIndex.DBConfig('localhost')
doc_index = QdrantDocumentIndex[MyDoc](qdrant_config)

# or just
doc_index = QdrantDocumentIndex[MyDoc](host='localhost')
```


**Creating an in-memory Qdrant document index**
```python
qdrant_config = QdrantDocumentIndex.DBConfig(location=":memory:")
doc_index = QdrantDocumentIndex[MyDoc](qdrant_config)
```

**Connecting to Qdrant Cloud service**
```python
qdrant_config = QdrantDocumentIndex.DBConfig(
    "https://YOUR-CLUSTER-URL.aws.cloud.qdrant.io",
    api_key="<your-api-key>",
)
doc_index = QdrantDocumentIndex[MyDoc](qdrant_config)
```

### Schema definition
In this code snippet, `QdrantDocumentIndex` takes a schema of the form of `MyDoc`.
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

The reason for this is that predefined documents don't hold information about the dimensionality of their `.embedding`
field. But this is crucial information for any vector database to work properly!

You can work around this problem by subclassing the predefined document and adding the dimensionality information:

=== "Using type hint"
    ```python
    from docarray.documents import TextDoc
    from docarray.typing import NdArray
    from docarray.index import QdrantDocumentIndex


    class MyDoc(TextDoc):
        embedding: NdArray[128]


    doc_index = QdrantDocumentIndex[MyDoc](host='localhost')
    ```

=== "Using Field()"
    ```python
    from docarray.documents import TextDoc
    from docarray.typing import AnyTensor
    from docarray.index import QdrantDocumentIndex
    from pydantic import Field


    class MyDoc(TextDoc):
        embedding: AnyTensor = Field(dim=128)


    doc_index = QdrantDocumentIndex[MyDoc](host='localhost')
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

That call to `index()` stores all documents in `docs` in the Document Index,
ready to be retrieved in the next step.

As you can see, `DocList[MyDoc]` and `QdrantDocumentIndex[MyDoc]` both have `MyDoc` as a parameter.
This means that they share the same schema, and in general, both the Document Index and the data that you want to store need to have compatible schemas.

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
The query should follow the [query language of Qdrant](https://qdrant.tech/documentation/concepts/filtering/).

In the following example let's filter for all the books that are cheaper than 29 dollars:

```python
from docarray import BaseDoc, DocList
from qdrant_client.http import models as rest


class Book(BaseDoc):
    title: str
    price: int


books = DocList[Book]([Book(title=f'title {i}', price=i * 10) for i in range(10)])
book_index = QdrantDocumentIndex[Book]()
book_index.index(books)

# filter for books that are cheaper than 29 dollars
query = rest.Filter(must=[rest.FieldCondition(key='price', range=rest.Range(lt=29))])
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


doc_index = QdrantDocumentIndex[NewsDoc](host='localhost')
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

For example, you can build a hybrid serach query that performs range filtering, vector search and text search:

```python
class SimpleDoc(BaseDoc):
    tens: NdArray[10]
    num: int
    text: str


doc_index = QdrantDocumentIndex[SimpleDoc](host='localhost')
index_docs = [
    SimpleDoc(
        id=f'{i}', tens=np.ones(10) * i, num=int(i / 2), text=f'Lorem ipsum {int(i/2)}'
    )
    for i in range(10)
]
doc_index.index(index_docs)

find_query = np.ones(10)
text_search_query = 'ipsum 1'
filter_query = rest.Filter(
    must=[
        rest.FieldCondition(
            key='num',
            range=rest.Range(
                gte=1,
                lt=5,
            ),
        )
    ]
)

query = (
    doc_index.build_query()
    .find(find_query, search_field='tens')
    .text_search(text_search_query, search_field='text')
    .filter(filter_query)
    .build(limit=5)
)

docs = doc_index.execute_query(query)
```


## Access documents

To access a document, you need to specify its `id`. You can also pass a list of `id`s to access multiple documents.

```python
# access a single Doc
doc_index[index_docs[16].id]

# access multiple Docs
doc_index[index_docs[16].id, index_docs[17].id]
```

## Delete documents

To delete documents, use the built-in function `del` with the `id` of the documents that you want to delete.
You can also pass a list of `id`s to delete multiple documents.

```python
# delete a single Doc
del doc_index[index_docs[16].id]

# delete multiple Docs
del doc_index[index_docs[17].id, index_docs[18].id]
```

## Update documents
In order to update a Document inside the index, you only need to re-index it with the updated attributes.

First, let's create a schema for our Document Index:
```python
import numpy as np
from docarray import BaseDoc, DocList
from docarray.typing import NdArray
from docarray.index import QdrantDocumentIndex


class MyDoc(BaseDoc):
    text: str
    embedding: NdArray[128]
```

Now, we can instantiate our Index and add some data:
```python
docs = DocList[MyDoc](
    [
        MyDoc(
            embedding=np.random.rand(10), text=f'I am the first version of Document {i}'
        )
        for i in range(100)
    ]
)
index = QdrantDocumentIndex[MyDoc]()
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

!!! tip "See all configuration options" To see all configuration options for the [QdrantDocumentIndex][docarray.index.backends.qdrant.QdrantDocumentIndex], you can do the following:

```python
from docarray.index import QdrantDocumentIndex

# the following can be passed to the __init__() method
db_config = QdrantDocumentIndex.DBConfig()
print(db_config)  # shows default values

# the following can be passed to the configure() method
runtime_config = QdrantDocumentIndex.RuntimeConfig()
print(runtime_config)  # shows default values
```

Note that the collection_name from the DBConfig is an Optional[str] with `None` as default value. This is because
the QdrantDocumentIndex will take the name the Document type that you use as schema. For example, for QdrantDocumentIndex[MyDoc](...) 
the data will be stored in a collection name MyDoc if no specific collection_name is passed in the DBConfig.


## Nested data and subindex search

The examples provided primarily operate on a basic schema where each field corresponds to a straightforward type such as `str` or `NdArray`. 
However, it is also feasible to represent and store nested documents in a Document Index, including scenarios where a document 
contains a `DocList` of other documents. 

Go to the [Nested Data](nested_data.md) section to learn more.
