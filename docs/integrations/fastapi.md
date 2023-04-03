# Use Docarray with FastAPI

You might already know that Docarray documents are Pydantic Models (with a twist), and as such they are fully compatible with FastAPI:

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


Further, you can send and receive lists of documents represented as a `DocArray` object:
> **Note**
> Currently, FastAPI receives DocArray objects as lists, so you have to construct a DocArray inside the function.
> Also, if you want to return DocArray, first you have to convert it to a list. 
> (shown in the example below)

```python
from typing import List

import numpy as np
from fastapi import FastAPI
from httpx import AsyncClient

from docarray import DocArray
from docarray.base_doc import DocArrayResponse
from docarray.documents import TextDoc

# Create a docarray
docs = DocArray[TextDoc]([TextDoc(text='first'), TextDoc(text='second')])

app = FastAPI()


# Always use our custom response class (needed to dump tensors)
@app.post("/doc/", response_class=DocArrayResponse)
async def create_embeddings(docs: List[TextDoc]) -> List[TextDoc]:
    # The docs FastAPI will receive will be treated as List[TextDoc]
    # so you need to cast it to DocArray
    docs = DocArray[TextDoc].construct(docs)

    # Embed docs
    for doc in docs:
        doc.embedding = np.zeros((3, 224, 224))

    # Return your DocArray as a list
    return list(docs)


async with AsyncClient(app=app, base_url="http://test") as ac:
    response = await ac.post("/doc/", data=docs.to_json())  # sending docs as json

assert response.status_code == 200
# You can read FastAPI's response in the following way
docs = DocArray[TextDoc].from_json(response.content.decode())
```
