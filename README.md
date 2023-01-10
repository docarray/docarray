# DocArray - Version 2

_**NOTE**: This introduction refers to version 2 of DocArray, a rewrite that is currently at alpha stage.
Not all features that are mentioned here are already implemented.
If you are looking for the version 2 implementation roadmap see [here](https://github.com/docarray/docarray/issues/780),
for the (already released) version 1 of DocArray
see [here](https://github.com/docarray/docarray)._

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
    image_tensor: Optional[TorchTensor[3, 224, 224]]
    embedding: Optional[TorchTensor[768]]


doc = MyDocument(
    description="This is a photo of a mountain",
    image_url="https://upload.wikimedia.org/wikipedia/commons/2/2f/Alpamayo.jpg",
)
doc.image_tensor = doc.image_url.load()  # load image tensor from URL
doc.embedding = CLIPImageEncoder()(
    doc.image_tensor
)  # create and store embedding using model of your choice
```

- **Model** data of any type (audio, video, text, images, 3D meshes, raw tensors, etc) as a single, unified data structure, the `Document`
  - A `Document` is a juiced-up [Pydantic Model](https://pydantic-docs.helpmanual.io/usage/models/), inheriting all the benefits, while extending it with ML focussed features 

### Use pre-defined `Document`s for common use cases:

```python
from docarray.documents import Image

doc = Image(
    url="https://upload.wikimedia.org/wikipedia/commons/2/2f/Alpamayo.jpg",
)
doc.image_tensor = doc.url.load()  # load image tensor from URL
doc.embedding = CLIPImageEncoder()(
    doc.image_tensor
)  # create and store embedding using model of your choice
```
### Compose nested Documents:

```python
from docarray import BaseDocument
from docarray.documents import Image, Text
import numpy as np


class MultiModalDocument(BaseDocument):
    image_doc: Image
    text_doc: Text


doc = MultiModalDocument(
    image_doc=Image(tensor=np.zeros((3, 224, 224))), text_doc=Text(text='hi!')
)
```

### Collect multiple `Documents` into a `DocumentArray`:
```python
from docarray import DocumentArray
from docarray.documents import Image

da = DocumentArray(
    [
        Image(
            url="https://upload.wikimedia.org/wikipedia/commons/2/2f/Alpamayo.jpg",
        )
        for _ in range(100)
    ]
)
```


## Send
- **Serialize** any `Document` or `DocumentArray` into _protobuf_, _json_, _jsonschema_, _bytes_ or _base64_
- Use in **microservice** architecture: Send over **HTTP** or **gRPC**
- Integrate seamlessly with **FastAPI** and **Jina**

```python
from docarray.documents import Image
from httpx import AsyncClient
import numpy as np

doc = Image(tensor=np.zeros((3, 224, 224)))

# JSON over HTTP
async with AsyncClient(app=app, base_url="http://test") as ac:
    response = await ac.post("/doc/", data=input_doc.json())

# (de)serialize from/to protobuf
Image.from_protobuf(doc.to_protobuf())
```

## Store
- Persist and `DocumentArray` using a **`DocumentStore`**
- Store your Documents in any supported (vector) database: **Elasticsearch**, **Qdrant**, **Weaviate**, **Redis**, **Milvus**, **ANNLite** or **SQLite**
- Leverage DocumentStores to **perform vector search on your multi-modal data**

```python
# NOTE: DocumentStores are not yet implemented in version 2
from docarray import DocumentArray
from docarray.documents import Image
from docarray.stores import DocumentStore
import numpy as np

da = DocumentArray([Image(embedding=np.zeros((128,))) for _ in range(1000)])
store = DocumentStore[Image](
    storage='qdrant'
)  # create a DocumentStore with Qdrant as backend
store.insert(da)  # insert the DocumentArray into the DocumentStore
# find the 10 most similar images based on the 'embedding' field
match = store.find(Image(embedding=np.zeros((128,))), field='embedding', top_k=10)
```

If you want to get a deeper understanding of DocArray v2, it is best to do so on the basis of your
use case and background:

## Coming from DocArray

If you are already using DocArray, you will be familiar with its [dataclass API](https://docarray.jina.ai/fundamentals/dataclass/).

_DocArray v2 is that idea, taken seriously._ Every `Document` is created through dataclass-like interface,
courtesy of [Pydantic](https://pydantic-docs.helpmanual.io/usage/models/).

This gives the following advantages:
- **Flexibility:** No need to conform to a fixed set of fields, your data defines the schema
- **Multi-modality:** Easily store multiple modalities and multiple embeddings in the same Document
- **Language agnostic:** At its core, Documents are just dictionaries. This makes it easy to create and send them from any language, not just Python.

## Coming from Pydantic

If you come from Pydantic, you can see Documents as juiced up models, and DocArray as a collection of goodies around them.

- **ML focussed types**: Tensor, TorchTensor, TFTensor, Embedding, ...
- **Types that are alive**: ImageUrl can `.load()` a URL to image tensor, TextUrl can load and tokenize text documents, etc.
- **Pre-built Documents** for different data modalities: Image, Text, 3DMesh, Video, Audio, ... Note that all of these will be valid Pydantic models!
- The concepts of **DocumentArray and DocumentStore**
- Cloud ready: Serialization to **Protobuf** for use with microservices and **gRPC**
- Support for **vector search functionalities**, such as `find()` and `embed()`

## Coming from FastAPI

Documents are Pydantic Models (with a twist), and as such they are fully compatible with FastAPI:

```python
import numpy as np
from fastapi import FastAPI
from httpx import AsyncClient

from docarray import BaseDocument
from docarray.documents import Image
from docarray.typing import NdArray


class InputDoc(BaseDocument):
    img: Image


class OutputDoc(BaseDocument):
    embedding_clip: NdArray
    embedding_bert: NdArray


input_doc = InputDoc(img=Image(tensor=np.zeros((3, 224, 224))))

app = FastAPI()


@app.post("/doc/", response_model=OutputDoc)
async def create_item(doc: InputDoc) -> OutputDoc:
    ## call my fancy model to generate the embeddings
    return OutputDoc(
        embedding_clip=np.zeros((100, 1)), embedding_bert=np.zeros((100, 1))
    )


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
```

## Coming from a vector database

If you came across docarray as a universal vector DB client, you can best think of it as **a new kind of ORM for vector databases**.

DocArray's job is to take multi-modal, nested and domain-specific data and to map it to a vector database,
store it there, and thus make it searchable:

```python
# NOTE: DocumentStores are not yet implemented in version 2
from docarray import DocumentArray, BaseDocument
from docarray.stores import DocumentStore
from docarray.documents import Image, Text
import numpy as np


class MyDoc(BaseDocument):
    image: Image
    text: Text
    description: str


def _random_my_doc():
    return MyDoc(
        image=Image(embedding=np.random.random((256,))),
        text=Text(embedding=np.random.random((128,))),
        description='this is a random document',
    )


da = DocumentArray([_random_my_doc() for _ in range(1000)])  # create some data
store = DocumentStore[MyDoc](
    storage='qdrant'
)  # create a DocumentStore with Qdrant as backend
store.insert(da)  # insert the DocumentArray into the DocumentStore

# find the 10 most similar images based on the image embedding field
match = store.find(
    Image(embedding=np.zeros((256,))), field='image__embedding', top_k=10
)
# find the 10 most similar images based on the image embedding field
match = store.find(Image(embedding=np.zeros((128,))), field='text__embedding', top_k=10)
```

## Install the alpha

to try out the alpha you can install it via git:
```shell
pip install "git+https://github.com/docarray/docarra@alphav2-0.1#egg=docarray[common,torch,image]"
```
or from the latest development branch
```shell
pip install "git+https://github.com/docarray/docarray@feat-rewrite-v2#egg=docarray[common,torch,image]"
```

## Further reading

- [V2 announcement blog post](https://github.com/docarray/notes/blob/main/blog/01-announcement.md)
- [Donation to Linux Foundation AI&Data blog post](https://jina.ai/news/donate-docarray-lf-for-inclusive-standard-multimodal-data-model/)
- [Submit ideas, feature requests, and discussions](https://github.com/docarray/docarray/discussions)
- ["Legacy" DocArray github page](https://github.com/docarray/docarray)
- ["Legacy" DocArray documentation](https://docarray.jina.ai/)

