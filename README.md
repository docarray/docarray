# DocArray - Version 2

> **Note**
> This introduction refers to version 2 of DocArray, a rewrite that is currently at the alpha stage.
> Not all features that are mentioned here are implemented yet.
> If you are looking for the version 2 implementation roadmap see [here](https://github.com/docarray/docarray/issues/780),
> for the (already released) version 1 of DocArray
> see [here](https://github.com/docarray/docarray)._

DocArray is a library for **representing, sending and storing multi-modal data**, with a focus on applications in **ML** and
**Neural Search**.

This means that DocArray lets you do the following things:

## Represent

```python
from docarray import BaseDocument
from docarray.typing import TorchTensor, ImageUrl
from typing import Optional


class MyDocument(BaseDocument):
    description: str
    image_url: ImageUrl
    image_tensor: Optional[TorchTensor[1704, 2272, 3]]
    # The field above only work with tensor of shape (1704, 2272, 3)
    embedding: Optional[TorchTensor]


doc = MyDocument(
    description="This is a photo of a mountain",
    image_url="https://upload.wikimedia.org/wikipedia/commons/2/2f/Alpamayo.jpg",
)
doc.image_tensor = doc.image_url.load()  # load image tensor from URL
```

```python
doc.embedding = clip_image_encoder(
    doc.image_tensor
)  # create and store embedding using model of your choice

print(doc.embedding.shape)
```

- **Model** data of any type (audio, video, text, images, 3D meshes, raw tensors, etc) as a Document, a single, unified data structure.
  - A `Document` is a juiced-up [Pydantic Model](https://pydantic-docs.helpmanual.io/usage/models/), inheriting all the benefits, while extending it with ML focused features.

### Use pre-defined `Document`s for common use cases:

```python
from docarray.documents import ImageDoc

doc = ImageDoc(
    url="https://upload.wikimedia.org/wikipedia/commons/2/2f/Alpamayo.jpg",
)
doc.tensor = doc.url.load()  # load image tensor from URL
doc.embedding = clip_image_encoder(
    doc.tensor
)  # create and store embedding using model of your choice
```
### Compose nested Documents:

```python
from docarray import BaseDocument
from docarray.documents import ImageDoc, TextDoc
import numpy as np


class MultiModalDocument(BaseDocument):
    image_doc: ImageDoc
    text_doc: TextDoc


doc = MultiModalDocument(
    image_doc=ImageDoc(tensor=np.zeros((3, 224, 224))), text_doc=TextDoc(text='hi!')
)
```

### Collect multiple `Documents` into a `DocumentArray`:
```python
from docarray import DocumentArray, BaseDocument
from docarray.typing import AnyTensor, ImageUrl
import numpy as np


class Image(BaseDocument):
    url: ImageUrl
    tensor: AnyTensor
```

```python
from docarray import DocumentArray

da = DocumentArray[Image](
    [
        Image(
            url="https://upload.wikimedia.org/wikipedia/commons/2/2f/Alpamayo.jpg",
            tensor=np.zeros((3, 224, 224)),
        )
        for _ in range(100)
    ]
)
```

Access fields at the DocumentArray level:

```python
print(len(da.tensor))
print(da.tensor[0].shape)
```

You can stack tensors if you want to perform in batch processing:

```python
da = da.stack()
```

```python
print(type(da.tensor))
print(da.tensor.shape)
```

## Send
- **Serialize** any `Document` or `DocumentArray` into _protobuf_, _json_, _jsonschema_, _bytes_ or _base64_
- Use in **microservice** architecture: Send over **HTTP** or **gRPC**
- Integrate seamlessly with **[FastAPI](https://github.com/tiangolo/fastapi/)** and **[Jina](https://github.com/jina-ai/jina/)**

```python
from docarray.documents import ImageDoc
from httpx import AsyncClient
import numpy as np

doc = ImageDoc(tensor=np.zeros((3, 224, 224)))

# JSON over HTTP
async with AsyncClient(app=app, base_url="http://test") as ac:
    response = await ac.post("/doc/", data=input_doc.json())
```

```python
# (de)serialize from/to protobuf
Image.from_protobuf(doc.to_protobuf())
```

## Store
- Persist a `DocumentArray` using a **`DocumentStore`**
- Store your Documents in any supported (vector) database: **Elasticsearch**, **Qdrant**, **Weaviate**, **Redis**, **Milvus**, **ANNLite** or **SQLite**
- Leverage DocumentStores to **perform vector search on your multi-modal data**

```python
# NOTE: DocumentStores are not yet implemented in version 2
from docarray import DocumentArray
from docarray.documents import ImageDoc
from docarray.stores import DocumentStore
import numpy as np

da = DocumentArray([ImageDoc(embedding=np.zeros((128,))) for _ in range(1000)])
store = DocumentStore[ImageDoc](
    storage='qdrant'
)  # create a DocumentStore with Qdrant as backend
store.insert(da)  # insert the DocumentArray into the DocumentStore
# find the 10 most similar images based on the 'embedding' field
match = store.find(ImageDoc(embedding=np.zeros((128,))), field='embedding', top_k=10)
```

If you want to get a deeper understanding of DocArray v2, it is best to do so on the basis of your
use case and background:

## Coming from DocArray

If you are already using DocArray, you will be familiar with its [dataclass API](https://docarray.jina.ai/fundamentals/dataclass/).

_DocArray v2 is that idea, taken seriously._ Every `Document` is created through dataclass-like interface,
courtesy of [Pydantic](https://pydantic-docs.helpmanual.io/usage/models/).

This gives the following advantages:
- **Flexibility:** No need to conform to a fixed set of fields -- your data defines the schema
- **Multi-modality:** Easily store multiple modalities and multiple embeddings in the same Document
- **Language agnostic:** At its core, Documents are just dictionaries. This makes it easy to create and send them from any language, not just Python.

## Coming from Pydantic

If you come from Pydantic, you can see Documents as juiced up models, and DocArray as a collection of goodies around them.

- **ML focused types**: Tensor, TorchTensor, TFTensor, Embedding, ...
- **Types that are alive**: ImageUrl can `.load()` a URL to image tensor, TextUrl can load and tokenize text documents, etc.
- **Pre-built Documents** for different data modalities: Image, Text, 3DMesh, Video, Audio and more. Note that all of these will be valid Pydantic models!
- The concepts of **DocumentArray and DocumentStore**
- Cloud-ready: Serialization to **Protobuf** for use with microservices and **gRPC**
- Support for **vector search functionalities**, such as `find()` and `embed()`

## Coming from PyTorch

DocArray can be used directly inside ML models to handle and represent multi-modal data. This allows you to reason about your data using DocArray's abstractions deep inside of `nn.Module`, and provides a (FastAPI-compatible) schema that eases the transition between model training and model serving.

To see the effect of this, let's first observe a vanilla PyTorch implementation of a tri-modal ML model:

```python
import torch
from torch import nn


class MyMultiModalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_encoder = AudioEncoder()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()

    def forward(self, text_1, text_2, image_1, image_2, audio_1, audio_2):
        embedding_text_1 = self.text_encoder(text_1)
        embedding_text_2 = self.text_encoder(text_2)

        embedding_image_1 = self.image_encoder(image_1)
        embedding_image_2 = self.image_encoder(image_2)

        embedding_audio_1 = self.image_encoder(audio_1)
        embedding_audio_2 = self.image_encoder(audio_2)

        return (
            embedding_text_1,
            embedding_text_2,
            embedding_image_1,
            embedding_image_2,
            embedding_audio_1,
            embedding_audio_2,
        )
```

Not very easy on the eyes if you ask us. And even worse, if you need to add one more modality you have to touch every part of your code base, changing the `forward()` return type and making a whole lot of changes downstream from that.

So, now let's see what the same code looks like with DocArray:

```python
from docarray import DocumentArray, BaseDocument
from docarray.documents import ImageDoc, TextDoc, AudioDoc
from docarray.typing import TorchTensor

import torch


class Podcast(BaseDocument):
    text: TextDoc
    image: ImageDoc
    audio: AudioDoc


class PairPodcast(BaseDocument):
    left: Podcast
    right: Podcast


class MyPodcastModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_encoder = AudioEncoder()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()

    def forward_podcast(self, da: DocumentArray[Podcast]) -> DocumentArray[Podcast]:
        da.audio.embedding = self.audio_encoder(da.audio.tensor)
        da.text.embedding = self.text_encoder(da.text.tensor)
        da.image.embedding = self.image_encoder(da.image.tensor)

        return da

    def forward(self, da: DocumentArray[PairPodcast]) -> DocumentArray[PairPodcast]:
        da.left = self.forward_podcast(da.left)
        da.right = self.forward_podcast(da.right)

        return da
```

Looks much better, doesn't it?
You instantly win in code readability and maintainability. And for the same price you can turn your PyTorch model into a FastAPI app and reuse your Document
schema definition (see below). Everything is handled in a pythonic manner by relying on type hints.

## Coming from TensorFlow

Similar to the PyTorch approach, you can also use DocArray with TensorFlow to handle and represent multi-modal data inside your ML model.

First off, to use DocArray with TensorFlow we first need to install it as follows:

```
pip install tensorflow==2.11.0
pip install protobuf==3.19.0
```

Compared to using DocArray with PyTorch, there is one main difference when using it with TensorFlow:\
While DocArray's `TorchTensor` is a subclass of `torch.Tensor`, this is not the case for the `TensorFlowTensor`: Due to some technical limitations of `tf.Tensor`, DocArray's `TensorFlowTensor` is not a subclass of `tf.Tensor` but rather stores a `tf.Tensor` in its `.tensor` attribute. 

How does this affect you? Whenever you want to access the tensor data to, let's say, do operations with it or hand it to your ML model, instead of handing over your `TensorFlowTensor` instance, you need to access its `.tensor` attribute.

This would look like the following:

```python
from typing import Optional

from docarray import DocumentArray, BaseDocument

import tensorflow as tf


class Podcast(BaseDocument):
    audio_tensor: Optional[AudioTensorFlowTensor]
    embedding: Optional[AudioTensorFlowTensor]


class MyPodcastModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.audio_encoder = AudioEncoder()

    def call(self, inputs: DocumentArray[Podcast]) -> DocumentArray[Podcast]:
        inputs.audio_tensor.embedding = self.audio_encoder(
            inputs.audio_tensor.tensor
        )  # access audio_tensor's .tensor attribute
        return inputs
```

## Coming from FastAPI

Documents are Pydantic Models (with a twist), and as such they are fully compatible with FastAPI:

```python
import numpy as np
from fastapi import FastAPI
from httpx import AsyncClient

from docarray import BaseDocument
from docarray.documents import ImageDoc
from docarray.typing import NdArray
from docarray.base_document import DocumentResponse


class InputDoc(BaseDocument):
    img: ImageDoc


class OutputDoc(BaseDocument):
    embedding_clip: NdArray
    embedding_bert: NdArray


input_doc = InputDoc(img=ImageDoc(tensor=np.zeros((3, 224, 224))))

app = FastAPI()


@app.post("/doc/", response_model=OutputDoc, response_class=DocumentResponse)
async def create_item(doc: InputDoc) -> OutputDoc:
    ## call my fancy model to generate the embeddings
    doc = OutputDoc(
        embedding_clip=np.zeros((100, 1)), embedding_bert=np.zeros((100, 1))
    )
    return doc


async with AsyncClient(app=app, base_url="http://test") as ac:
    response = await ac.post("/doc/", data=input_doc.json())
    resp_doc = await ac.get("/docs")
    resp_redoc = await ac.get("/redoc")
```

The big advantage here is **first-class support for ML centric data**, such as {Torch, TF, ...}Tensor, Embedding, etc.

This includes handy features such as validating the shape of a tensor:

```python
from docarray import BaseDocument
from docarray.typing import TorchTensor
import torch


class MyDoc(BaseDocument):
    tensor: TorchTensor[3, 224, 224]


doc = MyDoc(tensor=torch.zeros(3, 224, 224))  # works
doc = MyDoc(tensor=torch.zeros(224, 224, 3))  # works by reshaping
doc = MyDoc(tensor=torch.zeros(224))  # fails validation


class Image(BaseDocument):
    tensor: TorchTensor[3, 'x', 'x']


Image(tensor=torch.zeros(3, 224, 224))  # works
Image(
    tensor=torch.zeros(3, 64, 128)
)  # fails validation because second dimension does not match third
Image(
    tensor=torch.zeros(4, 224, 224)
)  # fails validation because of the first dimension
Image(
    tensor=torch.zeros(3, 64)
)  # fails validation because it does not have enough dimensions
```

## Coming from a vector database

If you came across DocArray as a universal vector database client, you can best think of it as **a new kind of ORM for vector databases**.

DocArray's job is to take multi-modal, nested and domain-specific data and to map it to a vector database,
store it there, and thus make it searchable:

```python
# NOTE: DocumentStores are not yet implemented in version 2
from docarray import DocumentArray, BaseDocument
from docarray.stores import DocumentStore
from docarray.documents import ImageDoc, TextDoc
import numpy as np


class MyDoc(BaseDocument):
    image: ImageDoc
    text: TextDoc
    description: str


def _random_my_doc():
    return MyDoc(
        image=ImageDoc(embedding=np.random.random((256,))),
        text=TextDoc(embedding=np.random.random((128,))),
        description='this is a random document',
    )


da = DocumentArray([_random_my_doc() for _ in range(1000)])  # create some data
store = DocumentStore[MyDoc](
    storage='qdrant'
)  # create a DocumentStore with Qdrant as backend
store.insert(da)  # insert the DocumentArray into the DocumentStore

# find the 10 most similar images based on the image embedding field
match = store.find(
    ImageDoc(embedding=np.zeros((256,))), field='image__embedding', top_k=10
)
# find the 10 most similar images based on the image embedding field
match = store.find(
    ImageDoc(embedding=np.zeros((128,))), field='text__embedding', top_k=10
)
```

## Enable logging

You can see more logs by setting the log level to `DEBUG` or `INFO`:

```python
from pydantic import Field
from docarray import BaseDocument
from docarray.index import HnswDocumentIndex
from docarray.typing import NdArray
import logging
# get the logger and set the log level to DEBUG
logging.getLogger('docarray').setLevel(logging.DEBUG)


# define a simple document and create a document index
class SimpleDoc(BaseDocument):
    vector: NdArray = Field(dim=10)


doc_store = HnswDocumentIndex[SimpleDoc](work_dir='temp_path/')
```

```console
INFO - docarray - DB config created
INFO - docarray - Runtime config created
DEBUG - docarray - Working directory set to temp_path/
WARNING - docarray - No index was created for `id` as it does not have a config
INFO - docarray - Created a new index for column `vector`
DEBUG - docarray - DB path set to temp_path/docs_sqlite.db
INFO - docarray - Connection to DB has been established
INFO - docarray - HnswDocumentIndex[SimpleDoc] has been initialized
```

## Install the alpha

To try out the alpha you can install it via git:

```shell
pip install "git+https://github.com/docarray/docarray@2023.01.18.alpha#egg=docarray[common,torch,image]"
```

...or from the latest development branch

```shell
pip install "git+https://github.com/docarray/docarray@feat-rewrite-v2#egg=docarray[common,torch,image]"
```

## See also

- [Join our Discord server](https://discord.gg/WaMp6PVPgR)
- [V2 announcement blog post](https://github.com/docarray/notes/blob/main/blog/01-announcement.md)
- [Donation to Linux Foundation AI&Data blog post](https://jina.ai/news/donate-docarray-lf-for-inclusive-standard-multimodal-data-model/)
- [Submit ideas, feature requests, and discussions](https://github.com/docarray/docarray/discussions)
- [v2 Documentation](https://docarray-v2--jina-docs.netlify.app/)
- ["Legacy" DocArray github page](https://github.com/docarray/docarray)
- ["Legacy" DocArray documentation](https://docarray.jina.ai/)
