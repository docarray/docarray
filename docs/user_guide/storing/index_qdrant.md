# Qdrant Document Index

!!! note "Install dependencies"
    To use [QdrantDocumentIndex][docarray.index.backends.qdrant.QdrantDocumentIndex], you need to install extra dependencies with the following command:

    ```console
    pip install "docarray[qdrant]"
    ```

The following is a starter script for using the [QdrantDocumentIndex][docarray.index.backends.qdrant.QdrantDocumentIndex],
based on the [Qdrant](https://qdrant.tech/) vector search engine.


## Basic Usage
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
docs = DocList[MyDoc](MyDoc(title=f'title #{i}', embedding=np.random.rand(128)) for i in range(10))

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

You can use docker-compose to create a local Qdrant service with the following `docker-compose.yml`.

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
docker-compose up
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

### Using a predefined Document as schema

DocArray offers a number of predefined Documents, like [ImageDoc][docarray.documents.ImageDoc] and [TextDoc][docarray.documents.TextDoc].
If you try to use these directly as a schema for a Document Index, you will get unexpected behavior:
Depending on the backend, an exception will be raised, or no vector index for ANN lookup will be built.

The reason for this is that predefined Documents don't hold information about the dimensionality of their `.embedding`
field. But this is crucial information for any vector database to work properly!

You can work around this problem by subclassing the predefined Document and adding the dimensionality information:

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

That call to `index()` stores all Documents in `docs` in the Document Index,
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

    In particular, this means that you can easily [index predefined Documents](#using-a-predefined-document-as-schema) into a Document Index.


## Vector Search

Now that you have indexed your data, you can perform vector similarity search using the [`find()`][docarray.index.abstract.BaseDocIndex.find] method.

By using a document of type `MyDoc`, [`find()`][docarray.index.abstract.BaseDocIndex.find], you can find
similar Documents in the Document Index:

=== "Search by Document"

    ```python
    # create a query Document
    query = MyDoc(embedding=np.random.rand(128), text='query')

    # find similar Documents
    matches, scores = doc_index.find(query, search_field='embedding', limit=5)

    print(f'{matches=}')
    print(f'{matches.text=}')
    print(f'{scores=}')
    ```

=== "Search by raw vector"

    ```python
    # create a query vector
    query = np.random.rand(128)

    # find similar Documents
    matches, scores = doc_index.find(query, search_field='embedding', limit=5)

    print(f'{matches=}')
    print(f'{matches.text=}')
    print(f'{scores=}')
    ```

To succesfully peform a vector search, you need to specify a `search_field`. This is the field that serves as the
basis of comparison between your query and the documents in the Document Index.

In this particular example you only have one field (`embedding`) that is a vector, so you can trivially choose that one.
In general, you could have multiple fields of type `NdArray` or `TorchTensor` or `TensorFlowTensor`, and you can choose
which one to use for the search.

The [`find()`][docarray.index.abstract.BaseDocIndex.find] method returns a named tuple containing the closest
matching documents and their associated similarity scores.

When searching on the subindex level, you can use the [`find_subindex()`][docarray.index.abstract.BaseDocIndex.find_subindex] method, which returns a named tuple containing the subindex documents, similarity scores and their associated root documents.

How these scores are calculated depends on the backend, and can usually be [configured](#configuration).

You can also search for multiple documents at once, in a batch, using the [find_batched()][docarray.index.abstract.BaseDocIndex.find_batched] method.

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
query = rest.Filter(
        must=[rest.FieldCondition(key='price', range=rest.Range(lt=29))]
    )
cheap_books = book_index.filter(filter_query=query)

assert len(cheap_books) == 3
for doc in cheap_books:
    doc.summary()
```

## Text Search

In addition to vector similarity search, the Document Index interface offers methods for text search:
[text_search()][docarray.index.abstract.BaseDocIndex.text_search],
as well as the batched version [text_search_batched()][docarray.index.abstract.BaseDocIndex.text_search_batched].

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


## Hybrid Search

Document Index supports atomic operations for vector similarity search, text search and filter search.

To combine these operations into a single, hybrid search query, you can use the query builder that is accessible
through [build_query()][docarray.index.abstract.BaseDocIndex.build_query]:

For example, you can build a hybrid serach query that performs range filtering, vector search and text search:

```python
class SimpleDoc(BaseDoc):
    tens: NdArray[10]
    num: int
    text: str


doc_index = QdrantDocumentIndex[SimpleDoc](host='localhost')
index_docs = [
    SimpleDoc(id=f'{i}', tens=np.ones(10) * i, num=int(i / 2), text=f'Lorem ipsum {int(i/2)}')
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

To access a document, you need to specify its `id`. You can also pass a list of `id` to access multiple documents.

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

## Update elements
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
    [MyDoc(embedding=np.random.rand(10), text=f'I am the first version of Document {i}') for i in range(100)]
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

Note that the collection_name from the DBConfig is an Optional[str] with None as default value. This is because
the QdrantDocumentIndex will take the name the Document type that you use as schema. For example, for QdrantDocumentIndex[MyDoc](...) 
the data will be stored in a collection name MyDoc if no specific collection_name is passed in the DBConfig.


## Nested data

The examples above all operate on a simple schema: All fields in `MyDoc` have "basic" types, such as `str` or `NdArray`.

**Index nested data:**

It is, however, also possible to represent nested Documents and store them in a Document Index.

In the following example you can see a complex schema that contains nested Documents.
The `YouTubeVideoDoc` contains a `VideoDoc` and an `ImageDoc`, alongside some "basic" fields:

```python
from docarray.typing import ImageUrl, VideoUrl, AnyTensor


# define a nested schema
class ImageDoc(BaseDoc):
    url: ImageUrl
    tensor: AnyTensor = Field(space='cosine', dim=64)


class VideoDoc(BaseDoc):
    url: VideoUrl
    tensor: AnyTensor = Field(space='cosine', dim=128)


class YouTubeVideoDoc(BaseDoc):
    title: str
    description: str
    thumbnail: ImageDoc
    video: VideoDoc
    tensor: AnyTensor = Field(space='cosine', dim=256)


# create a Document Index
doc_index = QdrantDocumentIndex[YouTubeVideoDoc](index_name='tmp2')

# create some data
index_docs = [
    YouTubeVideoDoc(
        title=f'video {i+1}',
        description=f'this is video from author {10*i}',
        thumbnail=ImageDoc(url=f'http://example.ai/images/{i}', tensor=np.ones(64)),
        video=VideoDoc(url=f'http://example.ai/videos/{i}', tensor=np.ones(128)),
        tensor=np.ones(256),
    )
    for i in range(8)
]

# index the Documents
doc_index.index(index_docs)
```

**Search nested data:**

You can perform search on any nesting level by using the dunder operator to specify the field defined in the nested data.

In the following example, you can see how to perform vector search on the `tensor` field of the `YouTubeVideoDoc` or on the `tensor` field of the nested `thumbnail` and `video` fields:

```python
# create a query Document
query_doc = YouTubeVideoDoc(
    title=f'video query',
    description=f'this is a query video',
    thumbnail=ImageDoc(url=f'http://example.ai/images/1024', tensor=np.ones(64)),
    video=VideoDoc(url=f'http://example.ai/videos/1024', tensor=np.ones(128)),
    tensor=np.ones(256),
)

# find by the `youtubevideo` tensor; root level
docs, scores = doc_index.find(query_doc, search_field='tensor', limit=3)

# find by the `thumbnail` tensor; nested level
docs, scores = doc_index.find(query_doc, search_field='thumbnail__tensor', limit=3)

# find by the `video` tensor; neseted level
docs, scores = doc_index.find(query_doc, search_field='video__tensor', limit=3)
```

### Nested data with subindex

Documents can be nested by containing a `DocList` of other documents, which is a slightly more complicated scenario than the one [above](#nested-data).

If a Document contains a DocList, it can still be stored in a Document Index.
In this case, the DocList will be represented as a new index (or table, collection, etc., depending on the database backend), that is linked with the parent index (table, collection, etc).

This still lets you index and search through all of your data, but if you want to avoid the creation of additional indexes you can refactor your document schemas without the use of DocLists.


**Index**

In the following example you can see a complex schema that contains nested Documents with subindex.
The `MyDoc` contains a `DocList` of `VideoDoc`, which contains a `DocList` of `ImageDoc`, alongside some "basic" fields:

```python
class ImageDoc(BaseDoc):
    url: ImageUrl
    tensor_image: AnyTensor = Field(space='cosine', dim=64)


class VideoDoc(BaseDoc):
    url: VideoUrl
    images: DocList[ImageDoc]
    tensor_video: AnyTensor = Field(space='cosine', dim=128)


class MediaDoc(BaseDoc):
    docs: DocList[VideoDoc]
    tensor: AnyTensor = Field(space='cosine', dim=256)


# create a Document Index
doc_index = QdrantDocumentIndex[MediaDoc](index_name='tmp3')

# create some data
index_docs = [
    MediaDoc(
        docs=DocList[VideoDoc](
            [
                VideoDoc(
                    url=f'http://example.ai/videos/{i}-{j}',
                    images=DocList[ImageDoc](
                        [
                            ImageDoc(
                                url=f'http://example.ai/images/{i}-{j}-{k}',
                                tensor_image=np.ones(64),
                            )
                            for k in range(10)
                        ]
                    ),
                    tensor_video=np.ones(128),
                )
                for j in range(10)
            ]
        ),
        tensor=np.ones(256),
    )
    for i in range(10)
]

# index the Documents
doc_index.index(index_docs)
```

**Search**

You can perform search on any subindex level by using `find_subindex()` method and the dunder operator `'root__subindex'` to specify the index to search on.

```python
# find by the `VideoDoc` tensor
root_docs, sub_docs, scores = doc_index.find_subindex(
    np.ones(128), subindex='docs', search_field='tensor_video', limit=3
)

# find by the `ImageDoc` tensor
root_docs, sub_docs, scores = doc_index.find_subindex(
    np.ones(64), subindex='docs__images', search_field='tensor_image', limit=3
)
```

