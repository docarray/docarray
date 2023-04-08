# Index
This section show you how to use the `DocArray.index` module. `DocArray.index` module is used to create index for the tensors so that one can search the document based on the vector similarity. `DocArray.index` implements the following index.

## Hnswlib

[HnswDocumentIndex][docarray.index.backends.hnswlib.HnswDocumentIndex] implement the index based on [hnswlib](https://github.com/nmslib/hnswlib). This is a lightweight implementation with vectors stored in memory.

!!! note
    To use [HnswDocumentIndex][docarray.index.backends.hnswlib.HnswDocumentIndex], one need to install the extra dependency with the following command

    ```console
    pip install "docarray[hnswlib]"
    ```

### Construct
To construct an index, you need to define the schema first. You can define the schema in the same way as define a `Doc`. The only difference is that you need to define the dimensionality of the vector space by `dim` and the name of the space by `space`. The `dim` argument must be an integer. The `space` argument can be one of `l2`, `ip` or `cosine`. TODO: add links to the detailed explaination

`work_dir` is the directory for storing the index. If there is an index in the directory, it will be automatically loaded. When the schema of the saved and the defined index do not match, an exception will be raised.

```python
from pydantic import Field

from docarray import BaseDoc
from docarray.index import HnswDocumentIndex
from docarray.typing import NdArray


class SimpleSchema(BaseDoc):
    tensor: NdArray = Field(dim=128, space='cosine')


doc_index = HnswDocumentIndex[SimpleSchema](work_dir='./tmp')
```

### Index
Use `.index()` to add `Doc` into the index. You need to define the `Doc` following the schema of the index. `.num_docs()` returns the total number of `Doc` in the index.

```python
from docarray import BaseDoc
from docarray.typing import NdArray
import numpy as np

class SimpleDoc(BaseDoc):
    tensor: NdArray

index_docs = [SimpleDoc(tensor=np.zeros(128)) for _ in range(64)]

doc_index.index(index_docs)
print(f'number of docs in the index: {doc_index.num_docs()}')
```

### Access
To access the `Doc`, you need to specify the `id`. You can also pass a list of `id` to access multiple `Doc`.

```python
# access a single Doc
doc_index[index_docs[16].id]

# access multiple Docs
doc_index[index_docs[16].id, index_docs[17].id]
```

### Delete
To delete the `Doc`, use the built-in function `del` with the `id` of the `Doc` to be deleted. You can also pass a list of `id` to delete multiple `Doc`.

```python
# delete a single Doc
del doc_index[index_docs[16].id]

# delete multiple Docs
del doc_index[index_docs[16].id, index_docs[17].id]
```

### Find nearest neighbors
Use `.find()` to find the nearest neighbors. You can use `limit` argument to configurate how much `Doc` to return.

```python
query = SimpleDoc(tensor=np.ones(10))

docs, scores = doc_index.find(query, limit=5)
```

### Nested index
When using the index, you can define multiple fields as well as the nested structure. In the following example, you have `YouTubeVideoDoc` including the `tensor` field calculated based on the description. Besides, `YouTbueVideoDoc` has `thumbnail` and `video` field, each of which has its own `tensor`.

```python
from docarray import BaseDoc
from docarray.typing import ImageUrl, VideoUrl, AnyTensor
from docarray.index import HnswDocumentIndex
import numpy as np
from pydantic import Field


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


doc_index = HnswDocumentIndex[YouTubeVideoDoc](work_dir='./tmp')
index_docs = [
    YouTubeVideoDoc(
        title=f'video {i+1}',
        description=f'this is video from author {10*i}',
        thumbnail=ImageDoc(
            url=f'http://example.ai/images/{i}',
            tensor=np.ones(64)),
        video=VideoDoc(
            url=f'http://example.ai/videos/{i}',
            tensor=np.ones(128)
        ),
        tensor=np.ones(256)
    ) for i in range(8)
]
doc_index.index(index_docs)
```

Use the `search_field` to specify which field to be used when performing the vector search. You can use the dunder operator to specify the field defined in the nested data. In the following codes, you can perform vector search on the `tensor` field of the `YouTubeVideoDoc` or on the `tensor` field of the `thumbnail` and `video` field. 

```python
# example of find nested and flat index
query_doc = YouTubeVideoDoc(
    title=f'video query',
    description=f'this is a query video',
    thumbnail=ImageDoc(
        url=f'http://example.ai/images/1024',
        tensor=np.ones(64)
    ),
    video=VideoDoc(
        url=f'http://example.ai/videos/1024',
        tensor=np.ones(128)
    ),
    tensor=np.ones(256)
)
# find by the youtubevideo tensor
docs, scores = doc_index.find(query_doc, search_field='tensor', limit=3)
# find by the thumbnail tensor
docs, scores = doc_index.find(query_doc, search_field='thumbnail__tensor', limit=3)
# find by the video tensor
docs, scores = doc_index.find(query_doc, search_field='video__tensor', limit=3)
```

To delete a nested data, you need to specify the `id`. 

!!! note
    You can only delete `Doc` at the top level. Deletion of the `Doc` on the lower level is not supported yet.

```python
# example of delete nested and flat index
del doc_index[index_docs[16].id, index_docs[32].id]
```