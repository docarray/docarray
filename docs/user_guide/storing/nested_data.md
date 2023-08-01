# Nested Data

Most of the examples you've seen operate on a simple schema: each field corresponds to a "basic" type, such as `str` or `NdArray`.

It is, however, also possible to represent nested documents and store them in a Document Index.

!!! note "Using a different vector database"
    In the following examples, we will use `InMemoryExactNNIndex` as our Document Index. 
    You can easily use Weaviate, Qdrant, Redis, Milvus or Elasticsearch instead -- their APIs are largely identical!
    To do so, check their respective documentation sections.

## Create and index
In the following example you can see a complex schema that contains nested documents.
The `YouTubeVideoDoc` contains a `VideoDoc` and an `ImageDoc`, alongside some "basic" fields:

```python
import numpy as np
from pydantic import Field

from docarray import BaseDoc, DocList
from docarray.index import InMemoryExactNNIndex
from docarray.typing import AnyTensor, ImageUrl, VideoUrl

# define a nested schema
class ImageDoc(BaseDoc):
    url: ImageUrl
    tensor: AnyTensor = Field(space='cosine_sim', dim=64)


class VideoDoc(BaseDoc):
    url: VideoUrl
    tensor: AnyTensor = Field(space='cosine_sim', dim=128)


class YouTubeVideoDoc(BaseDoc):
    title: str
    description: str
    thumbnail: ImageDoc
    video: VideoDoc
    tensor: AnyTensor = Field(space='cosine_sim', dim=256)


# create a Document Index
doc_index = InMemoryExactNNIndex[YouTubeVideoDoc]()

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

## Search

You can perform search on any nesting level by using the dunder operator to specify the field defined in the nested data.

In the following example, you can see how to perform vector search on the `tensor` field of the `YouTubeVideoDoc` or on the `tensor` field of the nested `thumbnail` and `video` fields:

```python
# create a query document
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

## Nested data with subindex search

Documents can be nested by containing a `DocList` of other documents, which is a slightly more complicated scenario than the one above.

If a document contains a `DocList`, it can still be stored in a Document Index.
In this case, the `DocList` will be represented as a new index (or table, collection, etc., depending on the database backend), that is linked with the parent index (table, collection, etc).

This still lets you index and search through all of your data, but if you want to avoid the creation of additional indexes you can refactor your document schemas without the use of `DocLists`.


### Index

In the following example, you can see a complex schema that contains nested `DocLists` of documents where we'll utilize subindex search.

The `MyDoc` contains a `DocList` of `VideoDoc`, which contains a `DocList` of `ImageDoc`, alongside some "basic" fields:

```python
class ImageDoc(BaseDoc):
    url: ImageUrl
    tensor_image: AnyTensor = Field(space='cosine_sim', dim=64)


class VideoDoc(BaseDoc):
    url: VideoUrl
    images: DocList[ImageDoc]
    tensor_video: AnyTensor = Field(space='cosine_sim', dim=128)


class MyDoc(BaseDoc):
    docs: DocList[VideoDoc]
    tensor: AnyTensor = Field(space='cosine_sim', dim=256)


# create a Document Index
doc_index = InMemoryExactNNIndex[MyDoc]()

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

### Search

You can perform search on any level by using `find_subindex()` method and the dunder operator `'root__subindex'` to specify the index to search on.

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