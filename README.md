<p align="center">
<img src="https://github.com/docarray/docarray/blob/main/docs/assets/logo-dark.svg?raw=true" alt="DocArray logo: The data structure for unstructured data" width="150px">
<br>
<b>The data structure for multimodal data</b>
</p>

<p align=center>
<a href="https://pypi.org/project/docarray/"><img src="https://img.shields.io/pypi/v/docarray?style=flat-square&amp;label=Release" alt="PyPI"></a>
<a href="https://bestpractices.coreinfrastructure.org/projects/6554"><img src="https://bestpractices.coreinfrastructure.org/projects/6554/badge"></a>
<a href="https://codecov.io/gh/docarray/docarray"><img alt="Codecov branch" src="https://img.shields.io/codecov/c/github/docarray/docarray/main?&logo=Codecov&logoColor=white&style=flat-square"></a>
<a href="https://pypistats.org/packages/docarray"><img alt="PyPI - Downloads from official pypistats" src="https://img.shields.io/pypi/dm/docarray?style=flat-square"></a>
<a href="https://discord.gg/WaMp6PVPgR"><img src="https://dcbadge.vercel.app/api/server/WaMp6PVPgR?theme=default-inverted&style=flat-square"></a>
</p>

> **Note**
> The README you're currently viewing is for DocArray>0.30, which introduces some significant changes from DocArray 0.21. If you wish to continue using the older DocArray <=0.21, ensure you install it via `pip install docarray==0.21`. Refer to its [codebase](https://github.com/docarray/docarray/tree/v0.21.0), [documentation](https://docarray.jina.ai), and [its hot-fixes branch](https://github.com/docarray/docarray/tree/docarray-v1-fixes) for more information.


DocArray is a Python library expertly crafted for the [representation](#represent), [transmission](#send), [storage](#store), and [retrieval](#retrieve) of multimodal data. Tailored for the development of multimodal AI applications, its design guarantees seamless integration with the extensive Python and machine learning ecosystems. As of January 2022, DocArray is openly distributed under the [Apache License 2.0](https://github.com/docarray/docarray/blob/main/LICENSE) and currently enjoys the status of a sandbox project within the [LF AI & Data Foundation](https://lfaidata.foundation/).



- :fire: Offers native support for **[NumPy](https://github.com/numpy/numpy)**, **[PyTorch](https://github.com/pytorch/pytorch)**, **[TensorFlow](https://github.com/tensorflow/tensorflow)**, and **[JAX](https://github.com/google/jax)**, catering specifically to **model training scenarios**.
- :zap: Based on **[Pydantic](https://github.com/pydantic/pydantic)**, and instantly compatible with web and microservice frameworks like **[FastAPI](https://github.com/tiangolo/fastapi/)** and **[Jina](https://github.com/jina-ai/jina/)**.
- :package: Provides support for vector databases such as **[Weaviate](https://weaviate.io/), [Qdrant](https://qdrant.tech/), [ElasticSearch](https://www.elastic.co/de/elasticsearch/), [Redis](https://redis.io/)**, and **[HNSWLib](https://github.com/nmslib/hnswlib)**.
- :chains: Allows data transmission as JSON over **HTTP** or as **[Protobuf](https://protobuf.dev/)** over **[gRPC](https://grpc.io/)**.

## Installation

To install DocArray from the CLI, run the following command:

```shell
pip install -U docarray
```

> **Note**
> To use DocArray <=0.21, make sure you install via `pip install docarray==0.21` and check out its [codebase](https://github.com/docarray/docarray/tree/v0.21.0) and [docs](https://docarray.jina.ai) and [its hot-fixes branch](https://github.com/docarray/docarray/tree/docarray-v1-fixes).

## Get Started
New to DocArray? Depending on your use case and background, there are multiple ways to learn about DocArray:
 
- [Coming from pure PyTorch or TensorFlow](#coming-from-pytorch)
- [Coming from Pydantic](#coming-from-pydantic)
- [Coming from FastAPI](#coming-from-fastapi)
- [Coming from a vector database](#coming-from-a-vector-database)
- [Coming from Langchain](#coming-from-langchain)


## Represent

DocArray empowers you to **represent your data** in a manner that is inherently attuned to machine learning.

This is particularly beneficial for various scenarios:

- :running: You are **training a model**: You're dealing with tensors of varying shapes and sizes, each signifying different elements. You desire a method to logically organize them.
- :cloud: You are **serving a model**: Let's say through FastAPI, and you wish to define your API endpoints precisely.
- :card_index_dividers: You are **parsing data**: Perhaps for future deployment in your machine learning or data science projects.

> :bulb: **Familiar with Pydantic?** You'll be pleased to learn
> that DocArray is not only constructed atop Pydantic but also maintains complete compatibility with it!
> Furthermore, we have a [specific section](#coming-from-pydantic) dedicated to your needs!

In essence, DocArray facilitates data representation in a way that mirrors Python dataclasses, with machine learning being an integral component:


```python
from docarray import BaseDoc
from docarray.typing import TorchTensor, ImageUrl
import torch


# Define your data model
class MyDocument(BaseDoc):
    description: str
    image_url: ImageUrl  # could also be VideoUrl, AudioUrl, etc.
    image_tensor: TorchTensor[1704, 2272, 3]  # you can express tensor shapes!


# Stack multiple documents in a Document Vector
from docarray import DocVec

vec = DocVec[MyDocument](
    [
        MyDocument(
            description="A cat",
            image_url="https://example.com/cat.jpg",
            image_tensor=torch.rand(1704, 2272, 3),
        ),
    ]
    * 10
)
print(vec.image_tensor.shape)  # (10, 1704, 2272, 3)
```

<details markdown="1">
  <summary>Click for more details</summary>

Let's take a closer look at how you can represent your data with DocArray:

```python
from docarray import BaseDoc
from docarray.typing import TorchTensor, ImageUrl
from typing import Optional
import torch


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

```python
# Create a document
doc = MyDocument(
    description="This is a photo of a mountain",
    image_url="https://upload.wikimedia.org/wikipedia/commons/2/2f/Alpamayo.jpg",
)

# Load image tensor from URL
doc.image_tensor = doc.image_url.load()


# Compute embedding with any model of your choice
def clip_image_encoder(image_tensor: TorchTensor) -> TorchTensor:  # dummy function
    return torch.rand(512)


doc.embedding = clip_image_encoder(doc.image_tensor)

print(doc.embedding.shape)  # torch.Size([512])
```

### Compose nested Documents

Of course, you can compose Documents into a nested structure:

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

You rarely work with a single data point at a time, especially in machine learning applications. That's why you can easily collect multiple `Documents`:

### Collect multiple `Documents`

When building or interacting with an ML system, usually you want to process multiple Documents (data points) at once.

DocArray offers two data structures for this:

- **`DocVec`**: A vector of `Documents`. All tensors in the documents are stacked into a single tensor. **Perfect for batch processing and use inside of ML models**.
- **`DocList`**: A list of `Documents`. All tensors in the documents are kept as-is. **Perfect for streaming, re-ranking, and shuffling of data**.

Let's take a look at them, starting with `DocVec`:

```python
from docarray import DocVec, BaseDoc
from docarray.typing import AnyTensor, ImageUrl
import numpy as np


class Image(BaseDoc):
    url: ImageUrl
    tensor: AnyTensor  # this allows torch, numpy, and tensor flow tensors


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

In the code snippet above, `DocVec` is **parametrized by the type of document** you want to use with it: `DocVec[Image]`.

This may look weird at first, but we're confident that you'll get used to it quickly!
Besides, it lets us do some cool things, like having **bulk access to the fields that you defined** in your document:

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

You can still bulk access the fields of your document:

```python
tensors = dl.tensor  # gets all the tensors in the DocList
print(type(tensors))  # as a list of tensors
print(dl.url)  # you can bulk access any other field, too
```

And you can insert, remove, and append documents to your `DocList`:

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
vec_2 = dl.to_doc_vec()
assert isinstance(vec_2, DocVec)

dl_2 = vec_2.to_doc_list()
assert isinstance(dl_2, DocList)
```

</details>

## Send

DocArray facilitates the **transmission of your data** in a manner inherently compatible with machine learning.

This includes native support for **Protobuf and gRPC**, along with **HTTP** and serialization to JSON, JSONSchema, Base64, and Bytes.

This feature proves beneficial for several scenarios:

- :cloud: You are **serving a model**, perhaps through frameworks like **[Jina](https://github.com/jina-ai/jina/)** or **[FastAPI](https://github.com/tiangolo/fastapi/)**
- :spider_web: You are **distributing your model** across multiple machines and need an efficient means of transmitting your data between nodes
- :gear: You are architecting a **microservice** environment and require a method for data transmission between microservices

> :bulb: **Are you familiar with FastAPI?** You'll be delighted to learn
> that DocArray maintains full compatibility with FastAPI!
> Plus, we have a [dedicated section](#coming-from-fastapi) specifically for you!

When it comes to data transmission, serialization is a crucial step. Let's delve into how DocArray streamlines this process:


```python
from docarray import BaseDoc
from docarray.typing import ImageTorchTensor
import torch


# model your data
class MyDocument(BaseDoc):
    description: str
    image: ImageTorchTensor[3, 224, 224]


# create a Document
doc = MyDocument(
    description="This is a description",
    image=torch.zeros((3, 224, 224)),
)

# serialize it!
proto = doc.to_protobuf()
bytes_ = doc.to_bytes()
json = doc.json()

# deserialize it!
doc_2 = MyDocument.from_protobuf(proto)
doc_4 = MyDocument.from_bytes(bytes_)
doc_5 = MyDocument.parse_raw(json)
```

Of course, serialization is not all you need. So check out how DocArray integrates with **[Jina](https://github.com/jina-ai/jina/)** and **[FastAPI](https://github.com/tiangolo/fastapi/)**.

## Store

After modeling and possibly distributing your data, you'll typically want to **store it** somewhere. That's where DocArray steps in!

**Document Stores** provide a seamless way to, as the name suggests, store your Documents. Be it locally or remotely, you can do it all through the same user interface:

- :cd: **On disk**, as a file in your local filesystem
- :bucket: On **[AWS S3](https://aws.amazon.com/de/s3/)**
- :cloud: On **[Jina AI Cloud](https://cloud.jina.ai/)**

The Document Store interface lets you push and pull Documents to and from multiple data sources, all with the same user interface.

For example, let's see how that works with on-disk storage:

```python
from docarray import BaseDoc, DocList


class SimpleDoc(BaseDoc):
    text: str


docs = DocList[SimpleDoc]([SimpleDoc(text=f'doc {i}') for i in range(8)])
docs.push('file://simple_docs')

docs_pull = DocList[SimpleDoc].pull('file://simple_docs')
```

## Retrieve

**Document Indexes** let you index your Documents in a **vector database** for efficient similarity-based retrieval.

This is useful for:

- :left_speech_bubble: Augmenting **LLMs and Chatbots** with domain knowledge ([Retrieval Augmented Generation](https://arxiv.org/abs/2005.11401))
- :mag: **Neural search** applications
- :bulb: **Recommender systems**

Currently, Document Indexes support **[Weaviate](https://weaviate.io/)**, **[Qdrant](https://qdrant.tech/)**, **[ElasticSearch](https://www.elastic.co/)**,  **[Redis](https://redis.io/)**, and **[HNSWLib](https://github.com/nmslib/hnswlib)**, with more to come!

The Document Index interface lets you index and retrieve Documents from multiple vector databases, all with the same user interface.

It supports ANN vector search, text search, filtering, and hybrid search.

```python
from docarray import DocList, BaseDoc
from docarray.index import HnswDocumentIndex
import numpy as np

from docarray.typing import ImageUrl, ImageTensor, NdArray


class ImageDoc(BaseDoc):
    url: ImageUrl
    tensor: ImageTensor
    embedding: NdArray[128]


# create some data
dl = DocList[ImageDoc](
    [
        ImageDoc(
            url="https://upload.wikimedia.org/wikipedia/commons/2/2f/Alpamayo.jpg",
            tensor=np.zeros((3, 224, 224)),
            embedding=np.random.random((128,)),
        )
        for _ in range(100)
    ]
)

# create a Document Index
index = HnswDocumentIndex[ImageDoc](work_dir='/tmp/test_index')


# index your data
index.index(dl)

# find similar Documents
query = dl[0]
results, scores = index.find(query, limit=10, search_field='embedding')
```


---

## Learn DocArray

Depending on your background and use case, there are different ways for you to understand DocArray.

### Coming from DocArray <=0.21

<details markdown="1">
  <summary>Click to expand</summary>

If you are using DocArray version 0.30.0 or lower, you will be familiar with its [dataclass API](https://docarray.jina.ai/fundamentals/dataclass/).

_DocArray >=0.30 is that idea, taken seriously._ Every document is created through a dataclass-like interface,
courtesy of [Pydantic](https://pydantic-docs.helpmanual.io/usage/models/).

This gives the following advantages:
- **Flexibility:** No need to conform to a fixed set of fields -- your data defines the schema
- **Multimodality:** At their core, documents are just dictionaries. This makes it easy to create and send them from any language, not just Python.

You may also be familiar with our old Document Stores for vector DB integration.
They are now called **Document Indexes** and offer the following improvements (see [here](#store) for the new API):

- **Hybrid search:** You can now combine vector search with text search, and even filter by arbitrary fields
- **Production-ready:** The new Document Indexes are a much thinner wrapper around the various vector DB libraries, making them more robust and easier to maintain
- **Increased flexibility:** We strive to support any configuration or setting that you could perform through the DB's first-party client

For now, Document Indexes support **[Weaviate](https://weaviate.io/)**, **[Qdrant](https://qdrant.tech/)**, **[ElasticSearch](https://www.elastic.co/)**, **[Redis](https://redis.io/)**,  Exact Nearest Neighbour search and **[HNSWLib](https://github.com/nmslib/hnswlib)**, with more to come.

</details>

### Coming from Pydantic

<details markdown="1">
  <summary>Click to expand</summary>

If you come from Pydantic, you can see DocArray documents as juiced up Pydantic models, and DocArray as a collection of goodies around them.

More specifically, we set out to **make Pydantic fit for the ML world** - not by replacing it, but by building on top of it!

This means you get the following benefits:

- **ML-focused types**: Tensor, TorchTensor, Embedding, ..., including **tensor shape validation**
- Full compatibility with **FastAPI**
- **DocList** and **DocVec** generalize the idea of a model to a _sequence_ or _batch_ of models. Perfect for **use in ML models** and other batch processing tasks.
- **Types that are alive**: ImageUrl can `.load()` a URL to image tensor, TextUrl can load and tokenize text documents, etc.
- Cloud-ready: Serialization to **Protobuf** for use with microservices and **gRPC**
- **Pre-built multimodal documents** for different data modalities: Image, Text, 3DMesh, Video, Audio and more. Note that all of these are valid Pydantic models!
- **Document Stores** and **Document Indexes** let you store your data and retrieve it using **vector search**

The most obvious advantage here is **first-class support for ML centric data**, such as `{Torch, TF, ...}Tensor`, `Embedding`, etc.

This includes handy features such as validating the shape of a tensor:

```python
from docarray import BaseDoc
from docarray.typing import TorchTensor
import torch


class MyDoc(BaseDoc):
    tensor: TorchTensor[3, 224, 224]


doc = MyDoc(tensor=torch.zeros(3, 224, 224))  # works
doc = MyDoc(tensor=torch.zeros(224, 224, 3))  # works by reshaping

try:
    doc = MyDoc(tensor=torch.zeros(224))  # fails validation
except Exception as e:
    print(e)
    # tensor
    # Cannot reshape tensor of shape (224,) to shape (3, 224, 224) (type=value_error)


class Image(BaseDoc):
    tensor: TorchTensor[3, 'x', 'x']


Image(tensor=torch.zeros(3, 224, 224))  # works

try:
    Image(
        tensor=torch.zeros(3, 64, 128)
    )  # fails validation because second dimension does not match third
except Exception as e:
    print()


try:
    Image(
        tensor=torch.zeros(4, 224, 224)
    )  # fails validation because of the first dimension
except Exception as e:
    print(e)
    # Tensor shape mismatch. Expected(3, 'x', 'x'), got(4, 224, 224)(type=value_error)

try:
    Image(
        tensor=torch.zeros(3, 64)
    )  # fails validation because it does not have enough dimensions
except Exception as e:
    print(e)
    # Tensor shape mismatch. Expected (3, 'x', 'x'), got (3, 64) (type=value_error)
```

</details>

### Coming from PyTorch

<details markdown="1">
  <summary>Click to expand</summary>

If you come from PyTorch, you can see DocArray mainly as a way of _organizing your data as it flows through your model_.

It offers you several advantages:

- Express **tensor shapes in type hints**
- **Group tensors that belong to the same object**, e.g. an audio track and an image
- **Go directly to deployment**, by re-using your data model as a [FastAPI](https://fastapi.tiangolo.com/) or [Jina](https://github.com/jina-ai/jina) API schema
- Connect model components between **microservices**, using Protobuf and gRPC

DocArray can be used directly inside ML models to handle and represent multimodaldata.
This allows you to reason about your data using DocArray's abstractions deep inside of `nn.Module`,
and provides a FastAPI-compatible schema that eases the transition between model training and model serving.

To see the effect of this, let's first observe a vanilla PyTorch implementation of a tri-modal ML model:

```python
import torch
from torch import nn


def encoder(x):
    return torch.rand(512)


class MyMultiModalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_encoder = encoder()
        self.image_encoder = encoder()
        self.text_encoder = encoder()

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
from torch import nn
import torch


def encoder(x):
    return torch.rand(512)


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
        self.audio_encoder = encoder()
        self.image_encoder = encoder()
        self.text_encoder = encoder()

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
schema definition (see [below](#coming-from-fastapi)). Everything is handled in a pythonic manner by relying on type hints.

</details>


### Coming from TensorFlow

<details markdown="1">
  <summary>Click to expand</summary>

Like the [PyTorch approach](#coming-from-pytorch), you can also use DocArray with TensorFlow to handle and represent multimodal data inside your ML model.

First off, to use DocArray with TensorFlow we first need to install it as follows:

```
pip install tensorflow==2.12.0
pip install protobuf==3.19.0
```

Compared to using DocArray with PyTorch, there is one main difference when using it with TensorFlow:
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

</details>

### Coming from FastAPI

<details markdown="1">
  <summary>Click to expand</summary>

Documents are Pydantic Models (with a twist), and as such they are fully compatible with FastAPI!

But why should you use them, and not the Pydantic models you already know and love?
Good question!

- Because of the ML-first features, types and validations, [here](#coming-from-pydantic)
- Because DocArray can act as an [ORM for vector databases](#coming-from-a-vector-database), similar to what SQLModel does for SQL databases

And to seal the deal, let us show you how easily documents slot into your FastAPI app:

```python
import numpy as np
from fastapi import FastAPI
from docarray.base_doc import DocArrayResponse
from docarray import BaseDoc
from docarray.documents import ImageDoc
from docarray.typing import NdArray


class InputDoc(BaseDoc):
    img: ImageDoc
    text: str


class OutputDoc(BaseDoc):
    embedding_clip: NdArray
    embedding_bert: NdArray


app = FastAPI()


def model_img(img: ImageTensor) -> NdArray:
    return np.zeros((100, 1))


def model_text(text: str) -> NdArray:
    return np.zeros((100, 1))


@app.post("/embed/", response_model=OutputDoc, response_class=DocArrayResponse)
async def create_item(doc: InputDoc) -> OutputDoc:
    doc = OutputDoc(
        embedding_clip=model_img(doc.img.tensor), embedding_bert=model_text(doc.text)
    )
    return doc


async with AsyncClient(app=app, base_url="http://test") as ac:
    response = await ac.post("/embed/", data=input_doc.json())
```

Just like a vanilla Pydantic model!

</details>

### Coming from a vector database

<details markdown="1">
  <summary>Click to expand</summary>

If you came across DocArray as a universal vector database client, you can best think of it as **a new kind of ORM for vector databases**.
DocArray's job is to take multimodal, nested and domain-specific data and to map it to a vector database,
store it there, and thus make it searchable:

```python
from docarray import DocList, BaseDoc
from docarray.index import HnswDocumentIndex
import numpy as np

from docarray.typing import ImageUrl, ImageTensor, NdArray


class ImageDoc(BaseDoc):
    url: ImageUrl
    tensor: ImageTensor
    embedding: NdArray[128]


# create some data
dl = DocList[ImageDoc](
    [
        ImageDoc(
            url="https://upload.wikimedia.org/wikipedia/commons/2/2f/Alpamayo.jpg",
            tensor=np.zeros((3, 224, 224)),
            embedding=np.random.random((128,)),
        )
        for _ in range(100)
    ]
)

# create a Document Index
index = HnswDocumentIndex[ImageDoc](work_dir='/tmp/test_index2')


# index your data
index.index(dl)

# find similar Documents
query = dl[0]
results, scores = index.find(query, limit=10, search_field='embedding')
```

Currently, DocArray supports the following vector databases:

- [Weaviate](https://www.weaviate.io/)
- [Qdrant](https://qdrant.tech/)
- [Elasticsearch](https://www.elastic.co/elasticsearch/) v8 and v7
- [Redis](https://redis.io/)
- ExactNNMemorySearch as a local alternative with exact kNN search.
- [HNSWlib](https://github.com/nmslib/hnswlib) as a local-first ANN alternative

An integration of [OpenSearch](https://opensearch.org/) is currently in progress.

DocArray <=0.21 also support [Milvus](https://milvus.io/), but this is not yet supported in the current version.

Of course this is only one of the things that DocArray can do, so we encourage you to check out the rest of this readme!

</details>


### Coming from Langchain

<details markdown="1">
  <summary>Click to expand</summary>

With DocArray, you can connect external data to LLMs through Langchain. DocArray gives you the freedom to establish 
flexible document schemas and choose from different backends for document storage.
After creating your document index, you can connect it to your Langchain app using [DocArrayRetriever](https://python.langchain.com/docs/modules/data_connection/retrievers/integrations/docarray_retriever).

Install Langchain via:
```shell
pip install langchain
```

1. Define a schema and create documents:
```python
from docarray import BaseDoc, DocList
from docarray.typing import NdArray
from langchain.embeddings.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# Define a document schema
class MovieDoc(BaseDoc):
    title: str
    description: str
    year: int
    embedding: NdArray[1536]


movies = [
    {"title": "#1 title", "description": "#1 description", "year": 1999},
    {"title": "#2 title", "description": "#2 description", "year": 2001},
]

# Embed `description` and create documents
docs = DocList[MovieDoc](
    MovieDoc(embedding=embeddings.embed_query(movie["description"]), **movie)
    for movie in movies
)
```

2. Initialize a document index using any supported backend:
```python
from docarray.index import (
    InMemoryExactNNIndex,
    HnswDocumentIndex,
    WeaviateDocumentIndex,
    QdrantDocumentIndex,
    ElasticDocIndex,
    RedisDocumentIndex,
)

# Select a suitable backend and initialize it with data
db = InMemoryExactNNIndex[MovieDoc](docs)
```

3. Finally, initialize a retriever and integrate it into your chain!
```python
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers import DocArrayRetriever


# Create a retriever
retriever = DocArrayRetriever(
    index=db,
    embeddings=embeddings,
    search_field="embedding",
    content_field="description",
)

# Use the retriever in your chain
model = ChatOpenAI()
qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)
```

Alternatively, you can use built-in vector stores. Langchain supports two vector stores: [DocArrayInMemorySearch](https://python.langchain.com/docs/modules/data_connection/vectorstores/integrations/docarray_in_memory) and [DocArrayHnswSearch](https://python.langchain.com/docs/modules/data_connection/vectorstores/integrations/docarray_hnsw). 
Both are user-friendly and are best suited to small to medium-sized datasets.

</details>


## See also

- [Documentation](https://docs.docarray.org)
- [DocArray<=0.21 documentation](https://docarray.jina.ai/)
- [Join our Discord server](https://discord.gg/WaMp6PVPgR)
- [Donation to Linux Foundation AI&Data blog post](https://jina.ai/news/donate-docarray-lf-for-inclusive-standard-multimodal-data-model/)
- [Roadmap](https://github.com/docarray/docarray/issues/1714)

> DocArray is a trademark of LF AI Projects, LLC
> 
