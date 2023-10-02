# Hnswlib Document Index

!!! note
    To use [HnswDocumentIndex][docarray.index.backends.hnswlib.HnswDocumentIndex], you need to install the extra dependency with the following command:

    ```console
    pip install "docarray[hnswlib]"
    ```


[HnswDocumentIndex][docarray.index.backends.hnswlib.HnswDocumentIndex] is a lightweight Document Index implementation
that runs fully locally and is best suited for small- to medium-sized datasets.
It stores vectors on disk in [hnswlib](https://github.com/nmslib/hnswlib), and stores all other data in [SQLite](https://www.sqlite.org/index.html).

!!! note "Production readiness"
    [HnswDocumentIndex][docarray.index.backends.hnswlib.HnswDocumentIndex] is a great starting point
    for small- to medium-sized datasets, but it is not battle tested in production. If scalability, uptime, etc. are
    important to you, we recommend you eventually transition to one of our database-backed Document Index implementations:

    - [QdrantDocumentIndex][docarray.index.backends.qdrant.QdrantDocumentIndex]
    - [WeaviateDocumentIndex][docarray.index.backends.weaviate.WeaviateDocumentIndex]
    - [ElasticDocumentIndex][docarray.index.backends.elastic.ElasticDocIndex]
    - [RedisDocumentIndex][docarray.index.backends.redis.RedisDocumentIndex]
    - [MilvusDocumentIndex][docarray.index.backends.milvus.MilvusDocumentIndex]

## Basic usage
This snippet demonstrates the basic usage of [HnswDocumentIndex][docarray.index.backends.hnswlib.HnswDocumentIndex]. It defines a document schema with a title and an embedding, 
creates ten dummy documents with random embeddings, initializes an instance of [HnswDocumentIndex][docarray.index.backends.hnswlib.HnswDocumentIndex] to index these documents, 
and performs a vector similarity search to retrieve the ten most similar documents to a given query vector.

```python
from docarray import BaseDoc, DocList
from docarray.index import HnswDocumentIndex
from docarray.typing import NdArray
import numpy as np

# Define the document schema.
class MyDoc(BaseDoc):
    title: str 
    embedding: NdArray[128]

# Create dummy documents.
docs = DocList[MyDoc](MyDoc(title=f'title #{i}', embedding=np.random.rand(128)) for i in range(10))

# Initialize a new HnswDocumentIndex instance and add the documents to the index.
doc_index = HnswDocumentIndex[MyDoc](work_dir='./tmp_0')
doc_index.index(docs)

# Perform a vector search.
query = np.ones(128)
retrieved_docs = doc_index.find(query, search_field='embedding', limit=10)
```

## Initialize

To create a Document Index, you first need a document class that defines the schema of your index:

```python
from docarray import BaseDoc
from docarray.index import HnswDocumentIndex
from docarray.typing import NdArray


class MyDoc(BaseDoc):
    embedding: NdArray[128]
    text: str


db = HnswDocumentIndex[MyDoc](work_dir='./tmp_1')
```

### Schema definition

In this code snippet, `HnswDocumentIndex` takes a schema of the form of `MyDoc`.
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
    from docarray.index import HnswDocumentIndex


    class MyDoc(TextDoc):
        embedding: NdArray[128]


    db = HnswDocumentIndex[MyDoc](work_dir='./tmp_2')
    ```

=== "Using Field()"
    ```python
    from docarray.documents import TextDoc
    from docarray.typing import AnyTensor
    from docarray.index import HnswDocumentIndex
    from pydantic import Field


    class MyDoc(TextDoc):
        embedding: AnyTensor = Field(dim=128)


    db = HnswDocumentIndex[MyDoc](work_dir='./tmp_3')
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
db.index(data)
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
db.index(docs)
```

That call to [`index()`][docarray.index.backends.hnswlib.HnswDocumentIndex.index] stores all Documents in `docs` in the Document Index,
ready to be retrieved in the next step.

As you can see, `DocList[MyDoc]` and `HnswDocumentIndex[MyDoc]` both have `MyDoc` as a parameter.
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

You can use the [`find()`][docarray.index.abstract.BaseDocIndex.find] function with a document of the type `MyDoc` 
to find similar documents within the Document Index:

=== "Search by Document"

    ```python
    # create a query document
    query = MyDoc(embedding=np.random.rand(128), text='query')

    # find similar documents
    matches, scores = db.find(query, search_field='embedding', limit=5)

    print(f'{matches=}')
    print(f'{matches.text=}')
    print(f'{scores=}')
    ```

=== "Search by raw vector"

    ```python
    # create a query vector
    query = np.random.rand(128)

    # find similar documents
    matches, scores = db.find(query, search_field='embedding', limit=5)

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

=== "Search by Documents"

    ```python
    # create some query Documents
    queries = DocList[MyDoc](
        MyDoc(embedding=np.random.rand(128), text=f'query {i}') for i in range(3)
    )

    # find similar documents
    matches, scores = db.find_batched(queries, search_field='embedding', limit=5)

    print(f'{matches=}')
    print(f'{matches[0].text=}')
    print(f'{scores=}')
    ```

=== "Search by raw vectors"

    ```python
    # create some query vectors
    query = np.random.rand(3, 128)

    # find similar documents
    matches, scores = db.find_batched(query, search_field='embedding', limit=5)

    print(f'{matches=}')
    print(f'{matches[0].text=}')
    print(f'{scores=}')
    ```

The [`find_batched()`][docarray.index.abstract.BaseDocIndex.find_batched] method returns a named tuple containing
a list of `DocList`s, one for each query, containing the closest matching documents and their similarity scores.


## Filter

You can filter your documents by using the `filter()` or `filter_batched()` method with a corresponding  filter query. 
The query should follow the query language of DocArray's [`filter_docs()`][docarray.utils.filter.filter_docs] function.

In the following example let's filter for all the books that are cheaper than 29 dollars:

```python
from docarray import BaseDoc, DocList


class Book(BaseDoc):
    title: str
    price: int


books = DocList[Book]([Book(title=f'title {i}', price=i * 10) for i in range(10)])
book_index = HnswDocumentIndex[Book](work_dir='./tmp_4')

# filter for books that are cheaper than 29 dollars
query = {'price': {'$lt': 29}}
cheap_books = book_index.filter(query)

assert len(cheap_books) == 3
for doc in cheap_books:
    doc.summary()
```



## Text search

!!! note
    The [HnswDocumentIndex][docarray.index.backends.hnswlib.HnswDocumentIndex] implementation does not support text search.

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
    year: int
    price: int
    embedding: NdArray[128]

# Create dummy documents.
docs = DocList[SimpleSchema](SimpleSchema(year=2000-i, price=i, embedding=np.random.rand(128)) for i in range(10))

doc_index = HnswDocumentIndex[SimpleSchema](work_dir='./tmp_5')
doc_index.index(docs)

query = (
    doc_index.build_query()  # get empty query object
    .filter(filter_query={'year': {'$gt': 1994}})  # pre-filtering
    .find(query=np.random.rand(128), search_field='embedding')  # add vector similarity search
    .filter(filter_query={'price': {'$lte': 3}})  # post-filtering
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
from docarray.index import HnswDocumentIndex
class MyDoc(BaseDoc):
    text: str
    embedding: NdArray[128]
```

Now, we can instantiate our Index and add some data:
```python
docs = DocList[MyDoc](
    [MyDoc(embedding=np.random.rand(128), text=f'I am the first version of Document {i}') for i in range(100)]
)
index = HnswDocumentIndex[MyDoc]()
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

This section lays out the configurations and options that are specific to [HnswDocumentIndex][docarray.index.backends.hnswlib.HnswDocumentIndex].

### DBConfig

The `DBConfig` of [HnswDocumentIndex][docarray.index.backends.hnswlib.HnswDocumentIndex] contains two argument:
`work_dir` and `default_column_configs`

`work_dir` is the location where all of the Index's data will be stored, namely the various HNSWLib indexes and the SQLite database.

You can pass this directly to the constructor:

```python
from docarray import BaseDoc
from docarray.index import HnswDocumentIndex
from docarray.typing import NdArray


class MyDoc(BaseDoc):
    embedding: NdArray[128]
    text: str


db = HnswDocumentIndex[MyDoc](work_dir='./tmp_6')
```

To load existing data, you can specify a directory that stores data from a previous session.

!!! note "Hnswlib file lock"
    Hnswlib uses a file lock to prevent multiple processes from accessing the same index at the same time.
    This means that if you try to open an index that is already open in another process, you will get an error.
    To avoid this, you can specify a different `work_dir` for each process.
    
`default_column_configs` contains the default mapping from Python types to column configurations.


You can see in the [section below](#field-wise-configurations) how to override configurations for specific fields.
If you want to set configurations globally, i.e. for all vector fields in your documents, you can do that using `DBConfig` or passing it at `__init__`:

```python
import numpy as np


db = HnswDocumentIndex[MyDoc](
    work_dir='./tmp_7',
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
    },
)
```

This will set the default configuration for all vector fields to the one specified in the example above.

!!! note
    Even if your vectors come from PyTorch or TensorFlow, you can (and should) still use the `np.ndarray` configuration.
    This is because all tensors are converted to `np.ndarray` under the hood.
    
!!! note
   max_elements is considered to have the initial maximum capacity of the index. However, the capacity of the index is doubled every time
   that the number of Documents in the index exceeds this capacity. Expanding the capacity is an expensive operation, therefore it can be important to
   choose an appropiate max_elements value at init time.

For more information on these settings, see [below](#field-wise-configurations).

Fields that are not vector fields (e.g. of type `str` or `int` etc.) do not offer any configuration, as they are simply
stored as-is in a SQLite database.

### Field-wise configuration

There are various setting that you can tweak for every vector field that you index into Hnswlib.

You pass all of those using the `field: Type = Field(...)` syntax:

```python
from pydantic import Field


class Schema(BaseDoc):
    tens: NdArray[100] = Field(max_elements=12, space='cosine')
    tens_two: NdArray[10] = Field(M=4, space='ip')


db = HnswDocumentIndex[Schema](work_dir='./tmp_8')
```

In the example above you can see how to configure two different vector fields, with two different sets of settings.

In this way, you can pass [all options that Hnswlib supports](https://github.com/nmslib/hnswlib#api-description):

| Keyword           | Description                                                                                                                                                                                                                                                                                                                                                                                                                            | Default |
|-------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|
| `max_elements`    | Maximum number of vector that can be stored                                                                                                                                                                                                                                                                                                                                                                                            | 1024    |
| `space`           | Vector space (distance metric) the index operates in. Supports 'l2', 'ip', and 'cosine'. <br/><span style="color:black">**Note:** In contrast to the other backends, for HnswDocumentIndex `'cosine'` refers to **cosine distance**, not cosine similarity. To transform one to the other, you can use: `cos_sim = 1 - cos_dist`. For more details see [here](https://en.wikipedia.org/wiki/Cosine_similarity#Cosine_Distance)</span>. | 'l2'    |
| `index`           | Whether or not an index should be built for this field.                                                                                                                                                                                                                                                                                                                                                                                | True    |
| `ef_construction` | defines a construction time/accuracy trade-off                                                                                                                                                                                                                                                                                                                                                                                         | 200     |
| `ef`              | parameter controlling query time/accuracy trade-off                                                                                                                                                                                                                                                                                                                                                                                    | 10      |
| `M`               | parameter that defines the maximum number of outgoing connections in the graph                                                                                                                                                                                                                                                                                                                                                         | 16      |
| `allow_replace_deleted`       | enables replacing of deleted elements with new added ones                                                                                                                                                                                                                                                                                                                                                                              | True    |
| `num_threads`  | sets the number of cpu threads to use                                                                                                                                                                                                                                                                                                                                                                                                  | 1       |

!!! note
    In HnswLibDocIndex  `space='cosine'` refers to cosine distance, not to cosine similarity, as it does for the other backends. 

You can find more details on the parameters [here](https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md).


### Database location

For `HnswDocumentIndex` you need to specify a `work_dir` where the data will be stored; for other backends you
usually specify a `host` and a `port` instead.

In addition to a host and a port, most backends can also take an `index_name`, `table_name`, `collection_name` or similar.
This specifies the name of the index/table/collection that will be created in the database.
You don't have to specify this though: By default, this name will be taken from the name of the Document type that you use as schema.
For example, for `WeaviateDocumentIndex[MyDoc](...)` the data will be stored in a Weaviate Class of name `MyDoc`.

In any case, if the location does not yet contain any data, we start from a blank slate.
If the location already contains data from a previous session, it will be accessible through the Document Index.



## Nested data and subindex search

The examples provided primarily operate on a basic schema where each field corresponds to a straightforward type such as `str` or `NdArray`. 
However, it is also feasible to represent and store nested documents in a Document Index, including scenarios where a document 
contains a `DocList` of other documents. 

Go to the [Nested Data](nested_data.md) section to learn more.
