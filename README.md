<p align="center">
<img src="https://github.com/docarray/docarray/blob/main/docs/_static/logo-light.svg?raw=true" alt="DocArray logo: The data structure for unstructured data" width="150px">
<br>
<b>The data structure for multimodal data</b>
</p>

<p align=center>
<a href="https://pypi.org/project/docarray/"><img src="https://img.shields.io/pypi/v/docarray?style=flat-square&amp;label=Release" alt="PyPI"></a>
<a href="https://codecov.io/gh/docarray/docarray"><img alt="Codecov branch" src="https://img.shields.io/codecov/c/github/docarray/docarray/main?logo=Codecov&logoColor=white&style=flat-square"></a>
<a href="https://bestpractices.coreinfrastructure.org/projects/6554"><img src="https://bestpractices.coreinfrastructure.org/projects/6554/badge"></a>
<a href="https://pypistats.org/packages/docarray"><img alt="PyPI - Downloads from official pypistats" src="https://img.shields.io/pypi/dm/docarray?style=flat-square"></a>
<a href="https://discord.gg/WaMp6PVPgR"><img src="https://dcbadge.vercel.app/api/server/WaMp6PVPgR?theme=default-inverted&style=flat-square"></a>
</p>

DocArray is a library for **representing, sending and storing multi-modal data**, perfect for **Machine Learning applications**.

DocArray handles your data while integrating seamlessly with the rest of your **Python and ML ecosystem**:

- :fire: DocArray has native compatibility for **NumPy**, **PyTorch** and **TensorFlow**, including for **model training use cases**
- :zap: DocArray is built on **Pydantic** and out-of-the-box compatible with **FastAPI**
- :package: DocArray can store data in vector databases such as **Weaviate, Qdrant, ElasticSearch** as well as **HNSWLib**
- :chains: DocArray data can be sent as JSON over **HTTP** or as **Protobuf** over **gRPC**

With that said, let's dig into the three pillars of DocArray:
1. [Represent](#represent)
2. [Send](#send)
3. [Store](#store)

> :bulb: **Where are you coming from?**: Depending on your use case and background, there are different was to "get" DocArray.
> You can navigate to the following section for an explanation that should fit your mindest:
> - [Coming from pure PyTorch or TensorFlow](#coming-from-torch-tf)
> - [Coming from Pydantic](#coming-from-pydantic)
> - [Coming from FastAPI](#coming-from-fastapi)
> - [Coming from a vector database](#coming-from-vector-database)


## Represent

DocArray allows you to **represent your data**, in a ML-native way.
This is useful for different use cases:
- :running_woman: You are **training a model**, there are myriads of tensors of different shapes and sizes flying around, representing different _things_, and you want to keep a straight head about them
- :cloud: You are **serving a model**, for example through FastAPI, and you want to specify your API endpoints
- :card_index_dividers: You are **parsing data** for later use in your ML or DS applications

> :bulb: **Coming from Pydantic?**: If you're currently using Pydantic for the use cases above, you should be happy to hear
> that DocArray is built on top of, and fully compatible with, Pydantic!
> Also, we have [dedicated section](#coming-from-pydantic) just for you!

Put simply, DocArray lets you represent your data in a dataclass-like way, with ML as a first class citizen:

```python
from docarray import BaseDoc
from docarray.typing import TorchTensor, ImageUrl

# Define your data model
class MyDocument(BaseDoc):
    description: str
    image_url: ImageUrl  # could also be VideoUrl, AudioUrl, etc.
    image_tensor: TorchTensor[1704, 2272, 3]  # you can express tensor shapes!


# Stack multiple documents
from docarray import DocVec

vec = DocVec[MyDocument](
    [
        MyDocument(
            description="A cat",
            image_url="https://example.com/cat.jpg",
            image_tensor=torch.rand(1704, 2272, 3),
        ),
    ]
    * 1000
)
print(vec.image_tensor.shape)  # (1000, 1704, 2272, 3)
```

<details>
  <summary>**Click for more details**</summary>

So let's take a closer look at how you can represent your data with DocArray:

```python
from docarray import BaseDoc
from docarray.typing import TorchTensor, ImageUrl
from typing import Optional


# Define your data model
class MyDocument(BaseDoc):
    description: str
    image_url: ImageUrl  # could also be VideoUrl, AudioUrl, etc.
    image_tensor: Optional[
        TorchTensor[1704, 2272, 3]
    ]  # could also be NdArray or TensorflowTensor
    embedding: Optional[TorchTensor]
```

So not only can you define the types of your data, you can even **specify the shape of your tensors!**

Once you have your model in form of a `Document`, you can work with it!

```python
# Create a document
doc = MyDocument(
    description="This is a photo of a mountain",
    image_url="https://upload.wikimedia.org/wikipedia/commons/2/2f/Alpamayo.jpg",
)

# Load image tensor from URL
doc.image_tensor = doc.image_url.load()

# Compute embedding with any model of your choice
doc.embedding = clip_image_encoder(doc.image_tensor)

print(doc.embedding.shape)
```

### Compose nested Documents

Of course you can compose Documents into a nested structure:

```python
from docarray import BaseDoc
from docarray.documents import ImageDoc, TextDoc
import numpy as np


class MultiModalDocument(BaseDoc):
    image_doc: ImageDoc
    text_doc: TextDoc


doc = MultiModalDocument(
    image_doc=ImageDoc(tensor=np.zeros((3, 224, 224))), text_doc=TextDoc(text='hi!')
)
```

Of course, you rarely work with a single data point at a time, especially in Machine Learning applications.

That's why you can easily collect multiple `Documents`:

### Collect multiple `Documents`

When building or interacting with an ML system, usually you want to process multiple Documents (data points) at once.

DocArray offers two data structures for this:
- **`DocVec`**: A vector of `Documents`. All tensors in the `Documents` are stacked up into a single tensor. **Perfect for batch processing and use inside of ML models**.
- **`DocList`**: A list of `Documents`. All tensors in the `Documents` are kept as-is. **Perfect for streaming, re-ranking, and shuffling of data**.

Let's take a look at them, starting with `DocVec`:

```python
from docarray import DocVec, BaseDoc
from docarray.typing import AnyTensor, ImageUrl
import numpy as np


class Image(BaseDoc):
    url: ImageUrl
    tensor: AnyTensor  # this allows torch, numpy, and tensorflow tensors


vec = DocVec[Image](  # the DocVec is parametrized by your personal schema!
    [
        Image(
            url="https://upload.wikimedia.org/wikipedia/commons/2/2f/Alpamayo.jpg",
            tensor=np.zeros((3, 224, 224)),
        )
        for _ in range(100)
    ]
)
```

As you can see in the code snippet above, `DocVec` is **parametrized by the type of Document** you want to use with it: `DocVec[Image]`.

This may look slightly weird at first, but we're confident that you'll get used to it quickly!
Besides, it allows us to do some cool things, like giving you **bulk access to the fields that you defined** in your `Document`:

```python
tensor = vec.tensor  # gets all the tensors in the DocVec
print(tensor.shape)  # which are stacked up into a single tensor!
print(vec.url)  # you can bulk access any other field, too
```

The second data structure, `DocList`, works in a similar way:

```python
from docarray import DocList

dl = DocList[Image](  # the DocList is parametrized by your personal schema!
    [
        Image(
            url="https://upload.wikimedia.org/wikipedia/commons/2/2f/Alpamayo.jpg",
            tensor=np.zeros((3, 224, 224)),
        )
        for _ in range(100)
    ]
)
```

You can still bulk access the fields of your `Document`:

```python
tensors = dl.tensor  # gets all the tensors in the DocVec
print(type(tensors))  # as a list of tensors
print(dl.url)  # you can bulk access any other field, too
```

And you can insert, remove, and append `Documents` to your `DocList`:

```python
# append
dl.append(
    Image(
        url="https://upload.wikimedia.org/wikipedia/commons/2/2f/Alpamayo.jpg",
        tensor=np.zeros((3, 224, 224)),
    )
)
# delete
del dl[0]
# insert
dl.insert(
    0,
    Image(
        url="https://upload.wikimedia.org/wikipedia/commons/2/2f/Alpamayo.jpg",
        tensor=np.zeros((3, 224, 224)),
    ),
)
```

And you can seamlessly switch between `DocVec` and `DocList`:

```python
vec_2 = dl.unstack()
assert isinstance(vec_2, DocVec)

dl_2 = vec_2.stack()
assert isinstance(dl_2, DocList)
```

</details>

## Send

- **Serialize** any `Document` or `DocArray` into _protobuf_, _json_, _jsonschema_, _bytes_ or _base64_
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
- Persist a `DocArray` using a **`DocumentStore`**
- Store your Documents in any supported (vector) database: **Elasticsearch**, **Qdrant**, **Weaviate**, **Redis**, **Milvus**, **ANNLite** or **SQLite**
- Leverage DocumentStores to **perform vector search on your multi-modal data**

```python
# NOTE: DocumentStores are not yet implemented in version 2
from docarray import DocList
from docarray.documents import ImageDoc
from docarray.stores import DocumentStore
import numpy as np

da = DocList([ImageDoc(embedding=np.zeros((128,))) for _ in range(1000)])
store = DocumentStore[ImageDoc](
    storage='qdrant'
)  # create a DocumentStore with Qdrant as backend
store.insert(da)  # insert the DocList into the DocumentStore
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
- The concepts of **DocArray and DocumentStore**
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
from docarray import DocList, BaseDoc
from docarray.documents import ImageDoc, TextDoc, AudioDoc
from docarray.typing import TorchTensor

import torch


class Podcast(BaseDoc):
    text: TextDoc
    image: ImageDoc
    audio: AudioDoc


class PairPodcast(BaseDoc):
    left: Podcast
    right: Podcast


class MyPodcastModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_encoder = AudioEncoder()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()

    def forward_podcast(self, docs: DocList[Podcast]) -> DocList[Podcast]:
        docs.audio.embedding = self.audio_encoder(docs.audio.tensor)
        docs.text.embedding = self.text_encoder(docs.text.tensor)
        docs.image.embedding = self.image_encoder(docs.image.tensor)

        return docs

    def forward(self, docs: DocList[PairPodcast]) -> DocList[PairPodcast]:
        docs.left = self.forward_podcast(docs.left)
        docs.right = self.forward_podcast(docs.right)

        return docs
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

from docarray import DocList, BaseDoc

import tensorflow as tf


class Podcast(BaseDoc):
    audio_tensor: Optional[AudioTensorFlowTensor]
    embedding: Optional[AudioTensorFlowTensor]


class MyPodcastModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.audio_encoder = AudioEncoder()

    def call(self, inputs: DocList[Podcast]) -> DocList[Podcast]:
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

from docarray import BaseDoc
from docarray.documents import ImageDoc
from docarray.typing import NdArray
from docarray.base_doc import DocumentResponse


class InputDoc(BaseDoc):
    img: ImageDoc


class OutputDoc(BaseDoc):
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
from docarray import BaseDoc
from docarray.typing import TorchTensor
import torch


class MyDoc(BaseDoc):
    tensor: TorchTensor[3, 224, 224]


doc = MyDoc(tensor=torch.zeros(3, 224, 224))  # works
doc = MyDoc(tensor=torch.zeros(224, 224, 3))  # works by reshaping
doc = MyDoc(tensor=torch.zeros(224))  # fails validation


class Image(BaseDoc):
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
from docarray import DocList, BaseDoc
from docarray.stores import DocumentStore
from docarray.documents import ImageDoc, TextDoc
import numpy as np


class MyDoc(BaseDoc):
    image: ImageDoc
    text: TextDoc
    description: str


def _random_my_doc():
    return MyDoc(
        image=ImageDoc(embedding=np.random.random((256,))),
        text=TextDoc(embedding=np.random.random((128,))),
        description='this is a random document',
    )


da = DocList([_random_my_doc() for _ in range(1000)])  # create some data
store = DocumentStore[MyDoc](
    storage='qdrant'
)  # create a DocumentStore with Qdrant as backend
store.insert(da)  # insert the DocArray into the DocumentStore

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
from docarray import BaseDoc
from docarray.index import HnswDocumentIndex
from docarray.typing import NdArray
import logging

# get the logger and set the log level to DEBUG
logging.getLogger('docarray').setLevel(logging.DEBUG)


# define a simple document and create a document index
class SimpleDoc(BaseDoc):
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
pip install "git+https://github.com/docarray/docarray@2023.01.18.alpha#egg=docarray[proto,torch,image]"
```

...or from the latest development branch

```shell
pip install "git+https://github.com/docarray/docarray@feat-rewrite-v2#egg=docarray[proto,torch,image]"
```

## See also

- [Join our Discord server](https://discord.gg/WaMp6PVPgR)
- [V2 announcement blog post](https://github.com/docarray/notes/blob/main/blog/01-announcement.md)
- [Donation to Linux Foundation AI&Data blog post](https://jina.ai/news/donate-docarray-lf-for-inclusive-standard-multimodal-data-model/)
- [Submit ideas, feature requests, and discussions](https://github.com/docarray/docarray/discussions)
- [v2 Documentation](https://docarray-v2--jina-docs.netlify.app/)
- ["Legacy" DocArray github page](https://github.com/docarray/docarray)
- ["Legacy" DocArray documentation](https://docarray.jina.ai/)
