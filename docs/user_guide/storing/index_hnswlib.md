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

## Basic Usage

To see how to create a [HnswDocumentIndex][docarray.index.backends.hnswlib.HnswDocumentIndex] instance, add Documents,
perform search, etc. see the [general user guide](./docindex.md).

## Configuration

This section lays out the configurations and options that are specific to [HnswDocumentIndex][docarray.index.backends.hnswlib.HnswDocumentIndex].

### DBConfig

The `DBConfig` of [HnswDocumentIndex][docarray.index.backends.hnswlib.HnswDocumentIndex] expects only one argument:
`work_dir`.

This is the location where all of the Index's data will be stored, namely the various HNSWLib indexes and the SQLite database.

You can pass this directly to the constructor:

```python
from docarray import BaseDoc
from docarray.index import HnswDocumentIndex
from docarray.typing import NdArray


class MyDoc(BaseDoc):
    embedding: NdArray[128]
    text: str


db = HnswDocumentIndex[MyDoc](work_dir='./path/to/db')
```

To load existing data, you can specify a directory that stores data from a previous session.

!!! note "Hnswlib file lock"
    Hnswlib uses a file lock to prevent multiple processes from accessing the same index at the same time.
    This means that if you try to open an index that is already open in another process, you will get an error.
    To avoid this, you can specify a different `work_dir` for each process.

### RuntimeConfig

The `RuntimeConfig` of [HnswDocumentIndex][docarray.index.backends.hnswlib.HnswDocumentIndex] contains only one entry:
the default mapping from Python types to column configurations.

You can see in the [section below](#field-wise-configurations) how to override configurations for specific fields.
If you want to set configurations globally, i.e. for all vector fields in your documents, you can do that using `RuntimeConfig`:

```python
import numpy as np

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
```

This will set the default configuration for all vector fields to the one specified in the example above.

!!! note
    Even if your vectors come from PyTorch or TensorFlow, you can (and should) still use the `np.ndarray` configuration.
    This is because all tensors are converted to `np.ndarray` under the hood.

For more information on these settings, see [below](#field-wise-configurations).

Fields that are not vector fields (e.g. of type `str` or `int` etc.) do not offer any configuration, as they are simply
stored as-is in a SQLite database.

### Field-wise configurations

There are various setting that you can tweak for every vector field that you index into Hnswlib.

You pass all of those using the `field: Type = Field(...)` syntax:

```python
from pydantic import Field


class Schema(BaseDoc):
    tens: NdArray[100] = Field(max_elements=12, space='cosine')
    tens_two: NdArray[10] = Field(M=4, space='ip')


db = HnswDocumentIndex[Schema](work_dir='/tmp/my_db')
```

In the example above you can see how to configure two different vector fields, with two different sets of settings.

In this way, you can pass [all options that Hnswlib supports](https://github.com/nmslib/hnswlib#api-description):

| Keyword           | Description                                                                                                                    | Default |
|-------------------|--------------------------------------------------------------------------------------------------------------------------------|---------|
| `max_elements`    | Maximum number of vector that can be stored                                                                                    | 1024    |
| `space`           | Vector space (similarity metric) the index operates in. Supports 'l2', 'ip', and 'cosine'                                      | 'l2'    |
| `index`           | Whether or not an index should be built for this field.                                                                        | True    |
| `ef_construction` | defines a construction time/accuracy trade-off | 200     |
| `ef`              | parameter controlling query time/accuracy trade-off | 10      |
| `M`               | parameter that defines the maximum number of outgoing connections in the graph | 16      |
| `allow_replace_deleted`       | enables replacing of deleted elements with new added ones | True    |
| `num_threads`  | sets the number of cpu threads to use | 1       |

You can find more details on the parameters [here](https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md).

## Nested Index

When using the index, you can define multiple fields and their nested structure. In the following example, you have `YouTubeVideoDoc` including the `tensor` field calculated based on the description. `YouTubeVideoDoc` has `thumbnail` and `video` fields, each with their own `tensor`.

```python
from docarray.typing import ImageUrl, VideoUrl, AnyTensor


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


doc_index = HnswDocumentIndex[YouTubeVideoDoc](work_dir='./tmp2')
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

You can use the `search_field` to specify which field to use when performing the vector search. You can use the dunder operator to specify the field defined in the nested data. In the following code, you can perform vector search on the `tensor` field of the `YouTubeVideoDoc` or on the `tensor` field of the `thumbnail` and `video` field:

```python
# example of find nested and flat index
query_doc = YouTubeVideoDoc(
    title=f'video query',
    description=f'this is a query video',
    thumbnail=ImageDoc(url=f'http://example.ai/images/1024', tensor=np.ones(64)),
    video=VideoDoc(url=f'http://example.ai/videos/1024', tensor=np.ones(128)),
    tensor=np.ones(256),
)
# find by the youtubevideo tensor
docs, scores = doc_index.find(query_doc, search_field='tensor', limit=3)
# find by the thumbnail tensor
docs, scores = doc_index.find(query_doc, search_field='thumbnail__tensor', limit=3)
# find by the video tensor
docs, scores = doc_index.find(query_doc, search_field='video__tensor', limit=3)
```

To delete nested data, you need to specify the `id`.

!!! note
    You can only delete `Doc` at the top level. Deletion of the `Doc` on lower levels is not yet supported.

```python
# example of deleting nested and flat index
del doc_index[index_docs[6].id]
```
