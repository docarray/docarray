# Introduction

A Document Index lets you store your Documents and search through them using vector similarity.

This is useful if you want to store a bunch of data, and at a later point retrieve documents that are similar to
some query that you provide.
Relevant concrete examples are neural search applications, augmenting LLMs and chatbots with domain knowledge ([Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)),
or recommender systems.

!!! question "How does vector similarity search work?"
    Without going into too much detail, the idea behind vector similarity search is the following:

    You represent every data point that you have (in our case, a document) as a _vector_, or _embedding_.
    This vector should represent as much semantic information about your data as possible: Similar data points should
    be represented by similar vectors.

    These vectors (embeddings) are usually obtained by passing the data through a suitable neural network that has been
    trained to produce such semantic representations - this is the _encoding_ step.

    Once you have your vectors that represent your data, you can store them, for example in a vector database.
    
    To perform similarity search, you take your input query and encode it in the same way as the data in your database.
    Then, the database will search through the stored vectors and return those that are most similar to your query.
    This similarity is measured by a _similarity metric_, which can be [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity),
    [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance), or any other metric that you can think of.

    If you store a lot of data, performing this similarity computation for every data point in your database is expensive.
    Therefore, vector databases usually perform _approximate nearest neighbor (ANN)_ search.
    There are various algorithms for doing this, such as [HNSW](https://arxiv.org/abs/1603.09320), but in a nutshell,
    they allow you to search through a large database of vectors very quickly, at the expense of a small loss in accuracy.

DocArray's Document Index concept achieves this by providing a unified interface to a number of [vector databases](https://learn.microsoft.com/en-us/semantic-kernel/concepts-ai/vectordb).
In fact, you can think of Document Index as an **[ORM](https://sqlmodel.tiangolo.com/db-to-code/) for vector databases**.

Currently, DocArray supports the following vector databases:

- [Weaviate](https://weaviate.io/)  |  [Docs](index_weaviate.md)
- [Qdrant](https://qdrant.tech/)  |  [Docs](index_qdrant.md)
- [Elasticsearch](https://www.elastic.co/elasticsearch/) v7 and v8  |  [Docs](index_elastic.md)
- [HNSWlib](https://github.com/nmslib/hnswlib)  |  [Docs](index_hnswlib.md)

For this user guide you will use the [HnswDocumentIndex][docarray.index.backends.hnswlib.HnswDocumentIndex]
because it doesn't require you to launch a database server. Instead, it will store your data locally.

!!! note "Using a different vector database"
    You can easily use Weaviate, Qdrant, or Elasticsearch instead -- they share the same API!
    To do so, check their respective documentation sections.

!!! note "Hnswlib-specific settings"
    The following sections explain the general concept of Document Index by using
    [HnswDocumentIndex][docarray.index.backends.hnswlib.HnswDocumentIndex] as an example.
    For HNSWLib-specific settings, check out the [HnswDocumentIndex][docarray.index.backends.hnswlib.HnswDocumentIndex] documentation
    [here](index_hnswlib.md).

## Create a Document Index

!!! note
    To use [HnswDocumentIndex][docarray.index.backends.hnswlib.HnswDocumentIndex], you need to install extra dependencies with the following command:

    ```console
    pip install "docarray[hnswlib]"
    ```

To create a Document Index, you first need a document that defines the schema of your index:

```python
from docarray import BaseDoc
from docarray.index import HnswDocumentIndex
from docarray.typing import NdArray


class MyDoc(BaseDoc):
    embedding: NdArray[128]
    text: str


db = HnswDocumentIndex[MyDoc](work_dir='./my_test_db')
```

### Schema definition

In this code snippet, `HnswDocumentIndex` takes a schema of the form of `MyDoc`.
The Document Index then _creates a column for each field in `MyDoc`_.

The column types in the backend database are determined by the type hints of the document's fields.
Optionally, you can [customize the database types for every field](#customize-configurations).

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
    from docarray.index import HnswDocumentIndex


    class MyDoc(TextDoc):
        embedding: NdArray[128]


    db = HnswDocumentIndex[MyDoc](work_dir='test_db')
    ```

=== "Using Field()"
    ```python
    from docarray.documents import TextDoc
    from docarray.typing import AnyTensor
    from docarray.index import HnswDocumentIndex
    from pydantic import Field


    class MyDoc(TextDoc):
        embedding: AnyTensor = Field(n_dim=128)


    db = HnswDocumentIndex[MyDoc](work_dir='test_db3')
    ```

Once the schema of your Document Index is defined in this way, the data that you are indexing can be either of the
predefined Document type, or your custom Document type.

The [next section](#index-data) goes into more detail about data indexing, but note that if you have some `TextDoc`s, `ImageDoc`s etc. that you want to index, you _don't_ need to cast them to `MyDoc`:

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
db.index(data)
```


**Database location:**

For `HnswDocumentIndex` you need to specify a `work_dir` where the data will be stored; for other backends you
usually specify a `host` and a `port` instead.

In addition to a host and a port, most backends can also take an `index_name`, `table_name`, `collection_name` or similar.
This specifies the name of the index/table/collection that will be created in the database.
You don't have to specify this though: By default, this name will be taken from the name of the Document type that you use as schema.
For example, for `WeaviateDocumentIndex[MyDoc](...)` the data will be stored in a Weaviate Class of name `MyDoc`.

In any case, if the location does not yet contain any data, we start from a blank slate.
If the location already contains data from a previous session, it will be accessible through the Document Index.

## Index data

Now that you have a Document Index, you can add data to it, using the [index()][docarray.index.abstract.BaseDocIndex.index] method:

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

As you can see, `DocList[MyDoc]` and `HnswDocumentIndex[MyDoc]` are both parameterized with `MyDoc`.
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

## Vector similarity search

Now that you have indexed your data, you can perform vector similarity search using the [find()][docarray.index.abstract.BaseDocIndex.find] method.

By using a document of type `MyDoc`, [find()][docarray.index.abstract.BaseDocIndex.find], you can find
similar Documents in the Document Index:

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

The [find()][docarray.index.abstract.BaseDocIndex.find] method returns a named tuple containing the closest
matching documents and their associated similarity scores.

When searching on subindex level, the `find()` method returns subindex documents. And the [find_subindex()][docarray.index.abstract.BaseDocIndex.find_subindex] method returns a named tuple containing the subindex documents, similarity scores and their associated root documents.

How these scores are calculated depends on the backend, and can usually be [configured](#customize-configurations).

### Batched search

You can also search for multiple documents at once, in a batch, using the [find_batched()][docarray.index.abstract.BaseDocIndex.find_batched] method.

=== "Search by Documents"

    ```python
    # create some query Documents
    queries = DocList[MyDoc](
        MyDoc(embedding=np.random.rand(128), text=f'query {i}') for i in range(3)
    )

    # find similar Documents
    matches, scores = db.find_batched(queries, search_field='embedding', limit=5)

    print(f'{matches=}')
    print(f'{matches[0].text=}')
    print(f'{scores=}')
    ```

=== "Search by raw vectors"

    ```python
    # create some query vectors
    query = np.random.rand(3, 128)

    # find similar Documents
    matches, scores = db.find_batched(query, search_field='embedding', limit=5)

    print(f'{matches=}')
    print(f'{matches[0].text=}')
    print(f'{scores=}')
    ```

The [find_batched()][docarray.index.abstract.BaseDocIndex.find_batched] method returns a named tuple containing
a list of `DocList`s, one for each query, containing the closest matching documents and their similarity scores.

## Filter search and text search

In addition to vector similarity search, the Document Index interface offers methods for text search and filtered search:
[text_search()][docarray.index.abstract.BaseDocIndex.text_search] and [filter()][docarray.index.abstract.BaseDocIndex.filter],
as well as their batched versions [text_search_batched()][docarray.index.abstract.BaseDocIndex.text_search_batched] and [filter_batched()][docarray.index.abstract.BaseDocIndex.filter_batched]. [filter_subindex()][docarray.index.abstract.BaseDocIndex.filter_subindex] if for filter on subindex level.

!!! note
    The [HnswDocumentIndex][docarray.index.backends.hnswlib.HnswDocumentIndex] implementation does not offer support for filter
    or text search.

    To see how to perform filter or text search, you can check out other backends that offer support.

## Hybrid search through the query builder

Document Index supports atomic operations for vector similarity search, text search and filter search.

To combine these operations into a single, hybrid search query, you can use the query builder that is accessible
through [build_query()][docarray.index.abstract.BaseDocIndex.build_query]:

```python
# prepare a query
q_doc = MyDoc(embedding=np.random.rand(128), text='query')

query = (
    db.build_query()  # get empty query object
    .find(query=q_doc, search_field='embedding')  # add vector similarity search
    .filter(filter_query={'text': {'$exists': True}})  # add filter search
    .build()  # build the query
)

# execute the combined query and return the results
results = db.execute_query(query)
print(f'{results=}')
```

In the example above you can see how to form a hybrid query that combines vector similarity search and filtered search
to obtain a combined set of results.

The kinds of atomic queries that can be combined in this way depends on the backend.
Some backends can combine text search and vector search, while others can perform filters and vectors search, etc.
To see what backend can do what, check out the [specific docs](#document-index).

## Access documents by `id`

To retrieve a document from a Document Index, you don't necessarily need to perform a fancy search.

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

## Delete Documents

In the same way you can access Documents by id, you can also delete them:

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

## Customize configurations

DocArray's philosophy is that each Document Index should "just work", meaning that it comes with a sane set of defaults
that get you most of the way there.

However, there are different configurations that you may want to tweak, including:

- The [ANN](https://ignite.apache.org/docs/latest/machine-learning/binary-classification/ann) algorithm used, for example [HNSW](https://www.pinecone.io/learn/hnsw/) or [ScaNN](https://ai.googleblog.com/2020/07/announcing-scann-efficient-vector.html)
- Hyperparameters of the ANN algorithm, such as `ef_construction` for HNSW
- The distance metric to use, such as cosine or L2 distance
- The data type of each column in the database
- And many more...

The specific configurations that you can tweak depend on the backend, but the interface to do so is universal.

Document Indexes differentiate between three different kind of configurations:

### Database configurations

_Database configurations_ are configurations that pertain to the entire database or table (as opposed to just a specific column),
and that you _don't_ dynamically change at runtime.

This commonly includes:

- host and port
- index or collection name
- authentication settings
- ...

For every backend, you can get a full list of configurations and their defaults:

```python
from docarray.index import HnswDocumentIndex


db_config = HnswDocumentIndex.DBConfig()
print(db_config)

# > HnswDocumentIndex.DBConfig(work_dir='.')
```

As you can see, `HnswDocumentIndex.DBConfig` is a dataclass that contains only one possible configuration, `work_dir`,
that defaults to `.`.

You can customize every field in this configuration:

=== "Pass individual settings"

    ```python
    db = HnswDocumentIndex[MyDoc](work_dir='/tmp/my_db')

    custom_db_config = db._db_config
    print(custom_db_config)

    # > HnswDocumentIndex.DBConfig(work_dir='/tmp/my_db')
    ```

=== "Pass entire configuration"

    ```python
    custom_db_config = HnswDocumentIndex.DBConfig(work_dir='/tmp/my_db')

    db = HnswDocumentIndex[MyDoc](custom_db_config)

    print(db._db_config)

    # > HnswDocumentIndex.DBConfig(work_dir='/tmp/my_db')
    ```

### Runtime configurations

_Runtime configurations_ are configurations that pertain to the entire database or table (as opposed to just a specific column),
and that you can dynamically change at runtime.


This commonly includes:
- default batch size for batching operations
- default mapping from pythong types to database column types
- default consistency level for various database operations
- ...

For every backend, you can get the full list of configurations and their defaults:

```python
from docarray.index import HnswDocumentIndex


runtime_config = HnswDocumentIndex.RuntimeConfig()
print(runtime_config)

# > HnswDocumentIndex.RuntimeConfig(default_column_config={<class 'numpy.ndarray'>: {'dim': -1, 'index': True, 'space': 'l2', 'max_elements': 1024, 'ef_construction': 200, 'ef': 10, 'M': 16, 'allow_replace_deleted': True, 'num_threads': 1}, None: {}})
```

As you can see, `HnswDocumentIndex.RuntimeConfig` is a dataclass that contains only one configuration:
`default_column_config`, which is a mapping from Python types to database column configurations.

You can customize every field in this configuration using the [configure()][docarray.index.abstract.BaseDocIndex.configure] method:

=== "Pass individual settings"

    ```python
    db = HnswDocumentIndex[MyDoc](work_dir='/tmp/my_db')

    db.configure(
        default_column_config={
            np.ndarray: {
                'dim': -1,
                'index': True,
                'space': 'ip',
                'max_elements': 2048,
                'ef_construction': 100,
                'ef': 15,
                'M': 8,
                'allow_replace_deleted': True,
                'num_threads': 5,
            },
            None: {},
        }
    )

    custom_runtime_config = db._runtime_config
    print(custom_runtime_config)

    # > HnswDocumentIndex.RuntimeConfig(default_column_config={<class 'numpy.ndarray'>: {'dim': -1, 'index': True, 'space': 'ip', 'max_elements': 2048, 'ef_construction': 100, 'ef': 15, 'M': 8, 'allow_replace_deleted': True, 'num_threads': 5}, None: {}})
    ```

=== "Pass entire configuration"

    ```python
    custom_runtime_config = HnswDocumentIndex.RuntimeConfig(
        default_column_config={
            np.ndarray: {
                'dim': -1,
                'index': True,
                'space': 'ip',
                'max_elements': 2048,
                'ef_construction': 100,
                'ef': 15,
                'M': 8,
                'allow_replace_deleted': True,
                'num_threads': 5,
            },
            None: {},
        }
    )

    db = HnswDocumentIndex[MyDoc](work_dir='/tmp/my_db')

    db.configure(custom_runtime_config)

    print(db._runtime_config)

    # > HHnswDocumentIndex.RuntimeConfig(default_column_config={<class 'numpy.ndarray'>: {'dim': -1, 'index': True, 'space': 'ip', 'max_elements': 2048, 'ef_construction': 100, 'ef': 15, 'M': 8, 'allow_replace_deleted': True, 'num_threads': 5}, None: {}})
    ```

After this change, the new setting will be applied to _every_ column that corresponds to a `np.ndarray` type.

### Column configurations

For many vector databases, individual columns can have different configurations.

This commonly includes:
- the data type of the column, e.g. `vector` vs `varchar`
- the dimensionality of the vector (if it is a vector column)
- whether an index should be built for a specific column

The available configurations vary from backend to backend, but in any case you can pass them
directly in the schema of your Document Index, using the `Field()` syntax:

```python
from pydantic import Field


class Schema(BaseDoc):
    tens: NdArray[100] = Field(max_elements=12, space='cosine')
    tens_two: NdArray[10] = Field(M=4, space='ip')


db = HnswDocumentIndex[Schema](work_dir='/tmp/my_db')
```

The `HnswDocumentIndex` above contains two columns which are configured differently:
- `tens` has a dimensionality of `100`, can take up to `12` elements, and uses the `cosine` similarity space
- `tens_two` has a dimensionality of `10`, and uses the `ip` similarity space, and an `M` hyperparameter of 4

All configurations that are not explicitly set will be taken from the `default_column_config` of the `RuntimeConfig`.

For an explanation of the configurations that are tweaked in this example, see the `HnswDocumentIndex` [documentation](index_hnswlib.md).

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
doc_index = HnswDocumentIndex[YouTubeVideoDoc](work_dir='/tmp2')

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

Documents can be nested by containing a `DocList` of other documents, which is a slightly more complicated scenario than the one [above][## Nested data].

If a Document contains a DocList, it can still be stored in a Document Index.
In this case, the DocList will be represented as a new index (or table, collection, etc., depending on the database backend), that is linked with the parent index (table, collection, ...).

This still lets index and search through all of your data, but if you want to avoid the creation of additional indexes you could try to refactor your document schemas without the use of DocList.


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


class MyDoc(BaseDoc):
    docs: DocList[VideoDoc]
    tensor: AnyTensor = Field(space='cosine', dim=256)


# create a Document Index
doc_index = HnswDocumentIndex[MyDoc](work_dir='/tmp3')

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

You can perform search on any subindex level by using the dunder operator `'subindex__field'` to specify the index to search on.

```python
# find by the `VideoDoc` tensor
sub_docs, scores = doc_index.find(
    np.ones(128), search_field='docs__tensor_video', limit=3
)  # return subindex docs
root_docs, sub_docs, scores = doc_index.find_subindex(
    np.ones(128), search_field='docs__tensor_video', limit=3
)  # return both root and subindex docs

# find by the `ImageDoc` tensor
sub_docs, scores = doc_index.find(
    np.ones(64), search_field='docs__images__tensor_image', limit=3
)  # return subindex docs
root_docs, sub_docs, scores = doc_index.find_subindex(
    np.ones(64), search_field='docs__images__tensor_image', limit=3
)  # return both root and subindex docs
```