# In-Memory Document Index


[InMemoryExactNNIndex][docarray.index.backends.in_memory.InMemoryExactNNIndex] stores all Documents in DocLists in memory. 
It is a great starting point for small datasets, where you may not want to launch a database server.

For vector search and filtering the InMemoryExactNNIndex utilizes DocArray's [`find()`][docarray.utils.find.find] and 
[`filter_docs()`][docarray.utils.filter.filter_docs] functions.

## Basic usage

To see how to create a [InMemoryExactNNIndex][docarray.index.backends.in_memory.InMemoryExactNNIndex] instance, add Documents,
perform search, etc. see the [general user guide](./docindex.md).

You can initialize the index as follows:

```python
from docarray import BaseDoc, DocList
from docarray.index.backends.in_memory import InMemoryExactNNIndex
from docarray.typing import NdArray


class MyDoc(BaseDoc):
    tensor: NdArray = None


docs = DocList[MyDoc](MyDoc() for _ in range(10))

doc_index = InMemoryExactNNIndex[MyDoc]()
doc_index.index(docs)

# or in one step, create with inserted docs.
doc_index = InMemoryExactNNIndex[MyDoc](docs)
```

Alternatively, you can pass an `index_file_path` argument to make sure that the index can be restored if persisted from that specific file.
```python
# Save your existing index as a binary file
docs = DocList[MyDoc](MyDoc() for _ in range(10))

doc_index = InMemoryExactNNIndex[MyDoc](index_file_path='docs.bin')
doc_index.index(docs)

# or in one step:
doc_index.persist()

# Initialize a new document index using the saved binary file
new_doc_index = InMemoryExactNNIndex[MyDoc](index_file_path='docs.bin')
```

## Configuration

This section lays out the configurations and options that are specific to [InMemoryExactNNIndex][docarray.index.backends.in_memory.InMemoryExactNNIndex].

The `DBConfig` of [InMemoryExactNNIndex][docarray.index.backends.in_memory.InMemoryExactNNIndex] contains two entries:
`index_file_path` and `default_column_mapping`, the default mapping from Python types to column configurations.

You can see in the [section below](#field-wise-configurations) how to override configurations for specific fields.
If you want to set configurations globally, i.e. for all vector fields in your Documents, you can do that using `DBConfig` or passing it at `__init__`::

```python
from collections import defaultdict
from docarray.typing import AbstractTensor
new_doc_index = InMemoryExactNNIndex[MyDoc](
    default_column_config=defaultdict(
        dict,
        {
            AbstractTensor: {'space': 'cosine_sim'},
        },
    )
)
```

This will set the default configuration for all vector fields to the one specified in the example above.

For more information on these settings, see [below](#field-wise-configurations).

Fields that are not vector fields (e.g. of type `str` or `int` etc.) do not offer any configuration.


### Field-wise configurations

For a vector field you can adjust the `space` parameter. It can be one of:

- `'cosine_sim'` (default)
- `'euclidean_dist'`
- `'sqeuclidean_dist'`

You pass it using the `field: Type = Field(...)` syntax:

```python
from docarray import BaseDoc
from pydantic import Field


class Schema(BaseDoc):
    tensor_1: NdArray[100] = Field(space='euclidean_dist')
    tensor_2: NdArray[100] = Field(space='sqeuclidean_dist')
```

In the example above you can see how to configure two different vector fields, with two different sets of settings.

## Nested index

When using the index, you can define multiple fields and their nested structure. In the following example, you have `YouTubeVideoDoc` including the `tensor` field calculated based on the description. `YouTubeVideoDoc` has `thumbnail` and `video` fields, each with their own `tensor`.

```python
import numpy as np
from docarray import BaseDoc
from docarray.index.backends.in_memory import InMemoryExactNNIndex
from docarray.typing import ImageUrl, VideoUrl, AnyTensor
from pydantic import Field


class ImageDoc(BaseDoc):
    url: ImageUrl
    tensor: AnyTensor = Field(space='cosine_sim')


class VideoDoc(BaseDoc):
    url: VideoUrl
    tensor: AnyTensor = Field(space='cosine_sim')


class YouTubeVideoDoc(BaseDoc):
    title: str
    description: str
    thumbnail: ImageDoc
    video: VideoDoc
    tensor: AnyTensor = Field(space='cosine_sim')


doc_index = InMemoryExactNNIndex[YouTubeVideoDoc]()
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
doc_index.index(index_docs)
```

## Search Documents

To search Documents, the `InMemoryExactNNIndex` uses DocArray's [`find`][docarray.utils.find.find] function.

You can use the `search_field` to specify which field to use when performing the vector search. 
You can use the dunder operator to specify the field defined in nested data. 
In the following code, you can perform vector search on the `tensor` field of the `YouTubeVideoDoc` 
or the `tensor` field of the `thumbnail` and `video` fields:

```python
# find by the youtubevideo tensor
query = parse_obj_as(NdArray, np.ones(256))
docs, scores = doc_index.find(query, search_field='tensor', limit=3)

# find by the thumbnail tensor
query = parse_obj_as(NdArray, np.ones(64))
docs, scores = doc_index.find(query, search_field='thumbnail__tensor', limit=3)

# find by the video tensor
query = parse_obj_as(NdArray, np.ones(128))
docs, scores = doc_index.find(query, search_field='video__tensor', limit=3)
```

## Filter Documents

To filter Documents, the `InMemoryExactNNIndex` uses DocArray's [`filter_docs()`][docarray.utils.filter.filter_docs] function.

You can filter your documents by using the `filter()` or `filter_batched()` method with a corresponding  filter query. 
The query should follow the query language of the DocArray's [`filter_docs()`][docarray.utils.filter.filter_docs] function.

In the following example let's filter for all the books that are cheaper than 29 dollars:

```python
from docarray import BaseDoc, DocList


class Book(BaseDoc):
    title: str
    price: int


books = DocList[Book]([Book(title=f'title {i}', price=i * 10) for i in range(10)])
book_index = InMemoryExactNNIndex[Book](books)

# filter for books that are cheaper than 29 dollars
query = {'price': {'$lte': 29}}
cheap_books = book_index.filter(query)

assert len(cheap_books) == 3
for doc in cheap_books:
    doc.summary()
```

<details>
    <summary>Output</summary>
    ```text
    📄 Book : 1f7da15 ...
    ╭──────────────────────┬───────────────╮
    │ Attribute            │ Value         │
    ├──────────────────────┼───────────────┤
    │ title: str           │ title 0       │
    │ price: int           │ 0             │
    ╰──────────────────────┴───────────────╯
    📄 Book : 63fd13a ...
    ╭──────────────────────┬───────────────╮
    │ Attribute            │ Value         │
    ├──────────────────────┼───────────────┤
    │ title: str           │ title 1       │
    │ price: int           │ 10            │
    ╰──────────────────────┴───────────────╯
    📄 Book : 49b21de ...
    ╭──────────────────────┬───────────────╮
    │ Attribute            │ Value         │
    ├──────────────────────┼───────────────┤
    │ title: str           │ title 2       │
    │ price: int           │ 20            │
    ╰──────────────────────┴───────────────╯
    ```
</details>

## Delete Documents 

To delete nested data, you need to specify the `id`.

!!! note
    You can only delete Documents at the top level. Deletion of Documents on lower levels is not yet supported.

```python
# example of deleting nested and flat index
del doc_index[index_docs[6].id]
```
