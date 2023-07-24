# Milvus Document Index

!!! note "Install dependencies"
    To use [MilvusDocumentIndex][docarray.index.backends.milvus.MilvusDocumentIndex], you need to install extra dependencies with the following command:

    ```console
    pip install "docarray[milvus]"
    ```

This is the user guide for the [MilvusDocumentIndex][docarray.index.backends.milvus.MilvusDocumentIndex],
focusing on special features and configurations of Milvus.


## Basic Usage
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

Now that you have a Document Index, you can add data to it, using the [index()][docarray.index.abstract.BaseDocIndex.index] method:

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

That call to [index()][docarray.index.backends.milvus.MilvusDocumentIndex.index] stores all Documents in `docs` into the Document Index,
ready to be retrieved in the next step.

As you can see, `DocList[MyDoc]` and `MilvusDocumentIndex[MyDoc]` are both parameterized with `MyDoc`.
This means that they share the same schema, and in general, the schema of a Document Index and the data that you want to store
need to have compatible schemas.

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

Now that you have indexed your data, you can perform vector similarity search using the [find()][docarray.index.abstract.BaseDocIndex.find] method.

By using a document of type `MyDoc`, [find()][docarray.index.abstract.BaseDocIndex.find], you can find
similar Documents in the Document Index:

=== "Search by Document"

    ```python
    # create a query Document
    query = MyDoc(embedding=np.random.rand(128), title='query')

    # find similar Documents
    matches, scores = doc_index.find(query, limit=5)

    print(f'{matches=}')
    print(f'{matches.title=}')
    print(f'{scores=}')
    ```

=== "Search by raw vector"

    ```python
    # create a query vector
    query = np.random.rand(128)

    # find similar Documents
    matches, scores = doc_index.find(query, limit=5)

    print(f'{matches=}')
    print(f'{matches.title=}')
    print(f'{scores=}')
    ```

To succesfully peform a vector search, you need to specify a `search_field`. This is the field that serves as the
basis of comparison between your query and the documents in the Document Index.

In this particular example you only have one field (`embedding`) that is a vector, so you can trivially choose that one.
In general, you could have multiple fields of type `NdArray` or `TorchTensor` or `TensorFlowTensor`, and you can choose
which one to use for the search.

The [find()][docarray.index.abstract.BaseDocIndex.find] method returns a named tuple containing the closest
matching documents and their associated similarity scores.

When searching on subindex level, you can use [find_subindex()][docarray.index.abstract.BaseDocIndex.find_subindex] method, which returns a named tuple containing the subindex documents, similarity scores and their associated root documents.

How these scores are calculated depends on the backend, and can usually be [configured](#customize-configurations).

You can also search for multiple documents at once, in a batch, using the [find_batched()][docarray.index.abstract.BaseDocIndex.find_batched] method.


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

## Text Search

In addition to vector similarity search, the Document Index interface offers methods for text search:
[text_search()][docarray.index.abstract.BaseDocIndex.text_search],
as well as the batched version [text_search_batched()][docarray.index.abstract.BaseDocIndex.text_search_batched].

!!! note
    The [MilvusDocumentIndex][docarray.index.backends.milvus.MilvusDocumentIndex] implementation does not offer support for text search.

    To see how to perform text search, you can check out other backends that offer support.


## Hybrid Search

Document Index supports atomic operations for vector similarity search, text search and filter search.

To combine these operations into a single, hybrid search query, you can use the query builder that is accessible
through [build_query()][docarray.index.abstract.BaseDocIndex.build_query]:

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
To see what backend can do what, check out the [specific docs](#document-index).


## Access Documents

To retrieve a document from a Document Index, you don't necessarily need to perform a fancy search.

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


## Delete Documents

In the same way you can access Documents by id, you can also delete them:

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
| `index_name`            | The name of the index in the Milvus database                                                                                  | None. Data will be stored in an index named after the Document type used as schema. |
| `user`              | The username for the Milvus server                                                                                            | None                                                                           |
| `password`              | The password for the Milvus server                                                                                            | None                                                                           |
| `token`              | Token for secure connection                                                                                            | ''                                                                             |
| `collection_description`              | Description of the collection in the database                                                                                            | ''                                                                             |
| `default_column_config` | The default configurations for every column type.                                                                            | dict                                                                           |

You can pass any of the above as keyword arguments to the `__init__()` method or pass an entire configuration object.


### Field-wise configurations


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
    tensor: AnyTensor = Field(is_embedding=True, space='cosine', dim=256)


# create a Document Index
doc_index = MilvusDocumentIndex[YouTubeVideoDoc](index_name='tmp2')

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
docs, scores = doc_index.find(query_doc, limit=3)
```

### Nested data with subindex

Documents can be nested by containing a `DocList` of other documents, which is a slightly more complicated scenario than the one [above](#nested-data).

If a Document contains a DocList, it can still be stored in a Document Index.
In this case, the DocList will be represented as a new index (or table, collection, etc., depending on the database backend), that is linked with the parent index (table, collection, ...).

This still lets index and search through all of your data, but if you want to avoid the creation of additional indexes you could try to refactor your document schemas without the use of DocList.


**Index**

In the following example you can see a complex schema that contains nested Documents with subindex.
The `MyDoc` contains a `DocList` of `VideoDoc`, which contains a `DocList` of `ImageDoc`, alongside some "basic" fields:

```python
class ImageDoc(BaseDoc):
    url: ImageUrl
    tensor_image: AnyTensor = Field(is_embedding=True, space='cosine', dim=64)


class VideoDoc(BaseDoc):
    url: VideoUrl
    images: DocList[ImageDoc]
    tensor_video: AnyTensor = Field(is_embedding=True, space='cosine', dim=128)


class MyDoc(BaseDoc):
    docs: DocList[VideoDoc]
    tensor: AnyTensor = Field(is_embedding=True, space='cosine', dim=256)


# create a Document Index
doc_index = MilvusDocumentIndex[MyDoc](index_name='tmp5')
doc_index.configure(MilvusDocumentIndex.RuntimeConfig(batch_size=10))


# create some data
index_docs = [
    MyDoc(
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
                            for k in range(5)
                        ]
                    ),
                    tensor_video=np.ones(128),
                )
                for j in range(5)
            ]
        ),
        tensor=np.ones(256),
    )
    for i in range(5)
]

# index the Documents
doc_index.index(index_docs)
```

**Search**

You can perform search on any subindex level by using `find_subindex()` method and the dunder operator `'root__subindex'` to specify the index to search on.

```python
# find by the `VideoDoc` tensor
root_docs, sub_docs, scores = doc_index.find_subindex(
    np.ones(128), subindex='docs', limit=3
)

# find by the `ImageDoc` tensor
root_docs, sub_docs, scores = doc_index.find_subindex(
    np.ones(64), subindex='docs__images', limit=3
)
```
