<p align="center">
<img src="https://github.com/docarray/docarray/blob/main/docs/assets/logo-dark.svg?raw=true" alt="DocArray logo: The data structure for unstructured data" width="150px">
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

> ⬆️ **DocArray v2**: This readme is for the second version of DocArray (starting at 0.30). If you want to use the older
> version (prior to 0.30) check out the [docarray-v1-fixes](https://github.com/docarray/docarray/tree/docarray-v1-fixes) branch

DocArray is a library for **representing, sending and storing multi-modal data**, perfect for **Machine Learning applications**.

With DocArray you can:

1. [**Represent data**](#represent)
2. [**Send data**](#send)
3. [**Store data**](#store)

DocArray handles your data while integrating seamlessly with the rest of your **Python and ML ecosystem**:

- :fire: Native compatibility for **[NumPy](https://github.com/numpy/numpy)**, **[PyTorch](https://github.com/pytorch/pytorch)** and **[TensorFlow](https://github.com/tensorflow/tensorflow)**, including for **model training use cases**
- :zap: Built on **[Pydantic](https://github.com/pydantic/pydantic)** and out-of-the-box compatible with **[FastAPI](https://github.com/tiangolo/fastapi/)** and **[Jina](https://github.com/jina-ai/jina/)**
- :package: Support for vector databases like **[Weaviate](https://weaviate.io/), [Qdrant](https://qdrant.tech/), [ElasticSearch](https://www.elastic.co/de/elasticsearch/)** and **[HNSWLib](https://github.com/nmslib/hnswlib)**
- :chains: Send data as JSON over **HTTP** or as **[Protobuf](https://protobuf.dev/)** over **[gRPC](https://grpc.io/)**

> :bulb: **Where are you coming from?** Based on your use case and background, there are different ways to understand DocArray:
> 
> - [Coming from pure PyTorch or TensorFlow](#coming-from-pytorch)
> - [Coming from Pydantic](#coming-from-pydantic)
> - [Coming from FastAPI](#coming-from-fastapi)
> - [Coming from a vector database](#coming-from-vector-database)

DocArray was released under the open-source [Apache License 2.0](https://github.com/docarray/docarray/blob/main/LICENSE) in January 2022. It is currently a sandbox project under [LF AI & Data Foundation](https://lfaidata.foundation/).

## Represent

DocArray allows you to **represent your data**, in an ML-native way.

This is useful for different use cases:

- :running: You are **training a model**: There are tensors of different shapes and sizes flying around, representing different _things_, and you want to keep a straight head about them.
- :cloud: You are **serving a model**: For example through FastAPI, and you want to specify your API endpoints.
- :card_index_dividers: You are **parsing data**: For later use in your ML or data science applications.

> :bulb: **Coming from Pydantic?** You should be happy to hear
> that DocArray is built on top of, and is fully compatible with, Pydantic!
> Also, we have a [dedicated section](#coming-from-pydantic) just for you!

Put simply, DocArray lets you represent your data in a dataclass-like way, with ML as a first class citizen:

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

Once you have your model in the form of a document, you can work with it!

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

DocArray allows you to **send your data** in an ML-native way.

This means there is native support for **Protobuf and gRPC**, on top of **HTTP** and serialization to JSON, JSONSchema, Base64, and Bytes.

This is useful for different use cases:

- :cloud: You are **serving a model**, for example through **[Jina](https://github.com/jina-ai/jina/)** or **[FastAPI](https://github.com/tiangolo/fastapi/)**
- :spider_web: You are **distributing your model** across machines and need to send your data between nodes
- :gear: You are building a **microservice** architecture and need to send your data between microservices

> :bulb: **Coming from FastAPI?** You should be happy to hear
> that DocArray is fully compatible with FastAPI!
> Also, we have a [dedicated section](#coming-from-fastapi) just for you!

Whenever you want to send your data, you need to serialize it, so let's take a look at how that works with DocArray:

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

Of course, serialization is not all you need. So check out how DocArray integrates with FastAPI and Jina.

## Store

Once you've modelled your data, and maybe sent it around, usually you want to **store it** somewhere.
DocArray has you covered!

**Document Stores** let you, well, store your Documents, locally or remotely, all with the same user interface:

- :cd: **On disk** as a file in your local file system
- :bucket: On **[AWS S3](https://aws.amazon.com/de/s3/)**
- :cloud: On **[Jina AI Cloud](https://cloud.jina.ai/)**

<details markdown="1">
  <summary>See Document Store usage</summary>

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
</details>

**Document Indexes** let you index your Documents in a **vector database** for efficient similarity-based retrieval.

This is useful for:

- :left_speech_bubble: Augmenting **LLMs and Chatbots** with domain knowledge ([Retrieval Augmented Generation](https://arxiv.org/abs/2005.11401))
- :mag: **Neural search** applications
- :bulb: **Recommender systems**

Currently, Document Indexes support **[Weaviate](https://weaviate.io/)**, **[Qdrant](https://qdrant.tech/)**, **[ElasticSearch](https://www.elastic.co/)**, and **[HNSWLib](https://github.com/nmslib/hnswlib)**, with more to come!

<details markdown="1">
  <summary>See Document Index usage</summary>

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

</details>

Depending on your background and use case, there are different ways for you to understand DocArray.

## Coming from old DocArray

<details markdown="1">
  <summary>Click to expand</summary>

If you are using DocArray version 0.30.0 or lower, you will be familiar with its [dataclass API](https://docarray.jina.ai/fundamentals/dataclass/).

_DocArray v2 is that idea, taken seriously._ Every document is created through a dataclass-like interface,
courtesy of [Pydantic](https://pydantic-docs.helpmanual.io/usage/models/).

This gives the following advantages:
- **Flexibility:** No need to conform to a fixed set of fields -- your data defines the schema
- **Multimodality:** At their core, documents are just dictionaries. This makes it easy to create and send them from any language, not just Python.

You may also be familiar with our old Document Stores for vector DB integration.
They are now called **Document Indexes** and offer the following improvements (see [here](#store) for the new API):

- **Hybrid search:** You can now combine vector search with text search, and even filter by arbitrary fields
- **Production-ready:** The new Document Indexes are a much thinner wrapper around the various vector DB libraries, making them more robust and easier to maintain
- **Increased flexibility:** We strive to support any configuration or setting that you could perform through the DB's first-party client

For now, Document Indexes support **[Weaviate](https://weaviate.io/)**, **[Qdrant](https://qdrant.tech/)**, **[ElasticSearch](https://www.elastic.co/)**, and **[HNSWLib](https://github.com/nmslib/hnswlib)**, with more to come.

</details>

## Coming from Pydantic

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

## Coming from PyTorch

<details markdown="1">
  <summary>Click to expand</summary>

If you come from PyTorch, you can see DocArray mainly as a way of _organizing your data as it flows through your model_.

It offers you several advantages:

- Express **tensor shapes in type hints**
- **Group tensors that belong to the same object**, e.g. an audio track and an image
- **Go directly to deployment**, by re-using your data model as a [FastAPI](https://fastapi.tiangolo.com/) or [Jina](https://github.com/jina-ai/jina) API schema
- Connect model components between **microservices**, using Protobuf and gRPC

DocArray can be used directly inside ML models to handle and represent multi-modal data.
This allows you to reason about your data using DocArray's abstractions deep inside of `nn.Module`,
and provides a FastAPI-compatible schema that eases the transition between model training and model serving.

To see the effect of this, let's first observe a vanilla PyTorch implementation of a tri-modal ML model:

```python
import torch
from torch import nn
import torch


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


## Coming from TensorFlow

<details markdown="1">
  <summary>Click to expand</summary>

Like the [PyTorch approach](#coming-from-pytorch), you can also use DocArray with TensorFlow to handle and represent multimodal data inside your ML model.

First off, to use DocArray with TensorFlow we first need to install it as follows:

```
pip install tensorflow==2.11.0
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

## Coming from FastAPI

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
from httpx import AsyncClient

from docarray import BaseDoc
from docarray.documents import ImageDoc
from docarray.typing import NdArray
from docarray.base_doc import DocArrayResponse


class InputDoc(BaseDoc):
    img: ImageDoc


class OutputDoc(BaseDoc):
    embedding_clip: NdArray
    embedding_bert: NdArray


input_doc = InputDoc(img=ImageDoc(tensor=np.zeros((3, 224, 224))))

app = FastAPI()


@app.post("/doc/", response_model=OutputDoc, response_class=DocArrayResponse)
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

Just like a vanilla Pydantic model!

</details>

## Coming from a vector database

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
- [HNSWlib](https://github.com/nmslib/hnswlib) as a local-first alternative

An integration of [OpenSearch](https://opensearch.org/) is currently in progress.

Legacy versions of DocArray also support [Redis](https://redis.io/) and [Milvus](https://milvus.io/), but these are not yet supported in the current version.

Of course this is only one of the things that DocArray can do, so we encourage you to check out the rest of this readme!

</details>


## Installation

To install DocArray from the CLI, run the following command:

```shell
pip install -U docarray
```

## See also

- [Documentation](https://docs.docarray.org)
- [Join our Discord server](https://discord.gg/WaMp6PVPgR)
- [Donation to Linux Foundation AI&Data blog post](https://jina.ai/news/donate-docarray-lf-for-inclusive-standard-multimodal-data-model/)
- ["Legacy" DocArray github page](https://github.com/docarray/docarray/tree/docarray-v1-fixes)
- ["Legacy" DocArray documentation](https://docarray-legacy.jina.ai/)

> DocArray is a trademark of LF AI Projects, LLC
