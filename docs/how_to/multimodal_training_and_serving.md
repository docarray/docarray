---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Multimodal deep learning with DocArray

DocArray is a library for representing, sending, and storing multimodal data that can be used for a variety of different
use cases.

Here we will focus on a workflow familiar to many ML engineers: Building and training a model, and then serving it to
users.

This document contains two parts:

1. **Representing**: We will use DocArray to represent multimodal data while **building and training a PyTorch model**.
We will see how DocArray can help to organize and group your modalities and tensors and make clear what methods to expect as inputs and return as outputs.
2. **Sending**: We will take the model that we built and trained in part one, and **serve it using FastAPI**.
We will see how DocArray narrows the gap between model development and model deployment, and how the same data models can be
reused in both contexts. That part will be very short, but that's the point!

So without further ado, let's dive into it!

## 1. Representing: Build and train a PyTorch model

We will train a [CLIP](https://arxiv.org/abs/2103.00020)-like model on a dataset composed of text-image pairs.
The goal is to obtain a model that can understand both text and images and project them into a common embedding space.

We train the CLIP-like model on the [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) dataset.
To run this, you need to download and unzip the data into the same folder as your code.

!!! note
    In this tutorial we do not aim to reproduce any CLIP results (our dataset is way too small anyway),
    but rather we want to show how DocArray data structures help researchers and practitioners write beautiful and 
    pythonic multimodal PyTorch code.

```bash
#!pip install "docarray[torch,image]"
#!pip install torchvision
#!pip install transformers
#!pip install fastapi
#!pip install pandas
```

```python
import itertools
from typing import Callable, Dict, List, Optional
```

```python
import docarray
import torch
```

```python
import torchvision
from torch import nn
from transformers import AutoTokenizer, DistilBertModel
```

```python
DEVICE = "cuda:0"  # change to your favourite device
```

<!-- #region tags=[] -->
### Create documents for handling multimodal data
<!-- #endregion -->

The first thing we want to achieve when using DocArray is to clearly model our data so that we never get confused
about which tensors represent what.

To do that we are using a concept that is at the core of DocArray: The document -- a collection of multimodal data.
The `BaseDoc` class allows users to define their own (nested, multimodal) document schema to represent any kind of complex data.

Let's start by defining a few documents to handle the different modalities that we will use during our training:

```python
from docarray import BaseDoc, DocList
from docarray.typing import TorchTensor, ImageUrl
```

Let's first create a document for our Text modality. It will contain a number of `Tokens`, which we also define:

```python
from docarray.documents import TextDoc as BaseText


class Tokens(BaseDoc):
    input_ids: TorchTensor[48]
    attention_mask: TorchTensor
```

```python
class Text(BaseText):
    tokens: Optional[Tokens]
```

Notice the [`TorchTensor`][docarray.typing.TorchTensor] type. It is a thin wrapper around `torch.Tensor` that can be used like any other Torch tensor, 
but also enables additional features. One such feature is shape parametrization (`TorchTensor[48]`), which lets you
hint and even enforce the desired shape of any tensor!

To represent our image data, we use DocArray's [`ImageDoc`][docarray.documents.ImageDoc]:

```python
from docarray.documents import ImageDoc
```

Under the hood, an `ImageDoc` looks something like this (with the only main difference that it can take tensors from any
supported ML framework):

```python
class ImageDoc(BaseDoc):
    url: Optional[ImageUrl]
    tensor: Optional[TorchTesor]
    embedding: Optional[TorchTensor]
```

Actually, the `BaseText` above also already includes `tensor`, `url` and `embedding` fields, so we can use those on our
`Text` document as well.

The final document used for training here is the `PairTextImage`, which simply combines the Text and Image modalities:

```python
class PairTextImage(BaseDoc):
    text: TextDoc
    image: ImageDoc
```

You then need to forward declare the following types. This will allow the objects to be properly pickled and unpickled.

This will be unnecessary once [this issue](https://github.com/docarray/docarray/issues/1330) is resolved.

```python
from docarray import DocVec
DocVec[Tokens]
DocVec[TextDoc]
DocVec[ImageDoc]
DocVec[PairTextImage]
```

### Create the dataset 

In this section we will create a multimodal pytorch dataset around the Flick8k dataset using DocArray.

We will use DocArray's data loading functionality to load the data and use Torchvision and Transformers to preprocess the data before feeding it to our deep learning model:

```python
from torch.utils.data import DataLoader, Dataset
```

```python
class VisionPreprocess:
    def __init__(self):
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(232),
                torchvision.transforms.RandomCrop(224),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, image: Image) -> None:
        image.tensor = self.transform(image.url.load())
```

```python
class TextPreprocess:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")

    def __call__(self, text: Text) -> None:
        assert isinstance(text, Text)
        text.tokens = Tokens(
            **self.tokenizer(
                text.text, padding="max_length", truncation=True, max_length=48
            )
        )
```

`VisionPreprocess` and `TextPreprocess` implement standard preprocessing steps for images and text, nothing special here.

```python
import pandas as pd


def get_flickr8k_da(file: str = "captions.txt", N: Optional[int] = None):
    df = pd.read_csv(file, nrows=N)
    da = DocList[PairTextImage](
        PairTextImage(text=Text(text=i.caption), image=Image(url=f"Images/{i.image}"))
        for i in df.itertuples()
    )
    return da
```

In the `get_flickr8k_da` method we process the [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) dataset into a `DocList`.

Now let's instantiate this dataset using the [`MultiModalDataset`][docarray.data.MultiModalDataset] class. The constructor takes in the `da` and a dictionary of preprocessing transformations:

```python
da = get_flickr8k_da()
preprocessing = {"image": VisionPreprocess(), "text": TextPreprocess()}
```

```python
from docarray.data import MultiModalDataset

dataset = MultiModalDataset[PairTextImage](da=da, preprocessing=preprocessing)
loader = DataLoader(
    dataset,
    batch_size=128,
    collate_fn=dataset.collate_fn,
    shuffle=True,
    num_workers=4,
    multiprocessing_context="fork",
)
```

### Create the Pytorch model that works on DocArray

In this section we will create two encoders, one per modality (Text and Image). These encoders are normal PyTorch `nn.Module`s.
The only difference is that they operate on `DocList` rather that on torch.Tensor:

```python
class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

    def forward(self, texts: DocList[TextDoc]) -> TorchTensor:
        last_hidden_state = self.bert(
            input_ids=texts.tokens.input_ids, attention_mask=texts.tokens.attention_mask
        ).last_hidden_state

        return self._mean_pool(last_hidden_state, texts.tokens.attention_mask)

    def _mean_pool(
        self, last_hidden_state: TorchTensor, attention_mask: TorchTensor
    ) -> TorchTensor:
        masked_output = last_hidden_state * attention_mask.unsqueeze(-1)
        return masked_output.sum(dim=1) / attention_mask.sum(-1, keepdim=True)
```

The `TextEncoder` takes a `DocList` of `TextDoc`s as input, and returns an embedding `TorchTensor` as output.
`DocList` can be seen as a list of `TextDoc` documents, and the encoder will treat it as one batch.

```python
class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True)
        self.linear = nn.LazyLinear(out_features=768)

    def forward(self, images: DocList[ImageDoc]) -> TorchTensor:
        x = self.backbone(images.tensor)
        return self.linear(x)
```

Similarly, the `VisionEncoder` also takes a `DocList` of `ImageDoc`s as input, and returns an embedding `TorchTensor` as output.
However, it operates on the `tensor` attribute of each document.

Now we can instantiate our encoders:

```python
vision_encoder = VisionEncoder().to(DEVICE)
text_encoder = TextEncoder().to(DEVICE)
```

As you can see, DocArray helps us clearly convey what data is expected as input and output for each method, all through Python type hints.

### Train the model in a contrastive way between Text and Image (CLIP)

Now that we have defined our dataloader and our models, we can train the two encoders is a contrastive way.
The goal is to match the representation of the text and the image for each pair in the dataset.

```python
optim = torch.optim.Adam(
    itertools.chain(vision_encoder.parameters(), text_encoder.parameters()), lr=3e-4
)
```

```python
def cosine_sim(x_mat: TorchTensor, y_mat: TorchTensor) -> TorchTensor:
    a_n, b_n = x_mat.norm(dim=1)[:, None], y_mat.norm(dim=1)[:, None]
    a_norm = x_mat / torch.clamp(a_n, min=1e-7)
    b_norm = y_mat / torch.clamp(b_n, min=1e-7)
    return torch.mm(a_norm, b_norm.transpose(0, 1)).squeeze()
```

```python
def clip_loss(image: DocList[Image], text: DocList[Text]) -> TorchTensor:
    sims = cosine_sim(image.embedding, text.embedding)
    return torch.norm(sims - torch.eye(sims.shape[0], device=DEVICE))
```

In the type hints of `cosine_sim` and `clip_loss` you can again notice that we can treat a `TorchTensor` like any other
`torch.Tensor`, and how we can make explicit what kind of data and data modalities the different functions expect.

```python
num_epoch = 1  # here you should do more epochs to really learn something
```

One thing to notice here is that our dataloader does not return a `torch.Tensor` but a `DocList[PairTextImage]`,
which is exactly what our model can operate on.

So let's write a training loop and train our encoders:

```python
from tqdm import tqdm

with torch.autocast(device_type="cuda", dtype=torch.float16):
    for epoch in range(num_epoch):
        for i, batch in tqdm(
            enumerate(loader), total=len(loader), desc=f"Epoch {epoch}"
        ):
            batch.to(DEVICE)  # DocList can be moved to device

            optim.zero_grad()
            # FORWARD PASS:
            batch.image.embedding = vision_encoder(batch.image)
            batch.text.embedding = text_encoder(batch.text)
            loss = clip_loss(batch.image, batch.text)
            if i % 30 == 0:
                print(f"{i+epoch} steps , loss : {loss}")
            loss.backward()
            optim.step()
```

Here we see how we can immediately group the output of each encoder with the document (and modality) it belong to.

And with all that, we've successfully trained a CLIP-like model without ever getting confused about the meaning of any tensors!

## 2. Sending: Serve the model using FastAPI

Now that we have a trained CLIP model, let's see how we can serve this model with a REST API by reusing most of the code above.

Let's use [FastAPI](https://fastapi.tiangolo.com/) for that!

FastAPI is powerful because it allows you to define your Rest API data schema in pure Python.
And DocArray is fully compatible with FastAPI and Pydantic, which means that as long as you have a function that takes a document as input, 
FastAPI will be able to automatically translate it into a fully fledged API with documentation, OpenAPI specification and more:

```python
from fastapi import FastAPI
from docarray.base_doc import DocumentResponse
```

```python
app = FastAPI()
```

```python
vision_encoder = vision_encoder.eval()
text_encoder = text_encoder.eval()
```

Now all we need to do is to tell FastAPI what methods it should use to serve the model:

```python
text_preprocess = TextPreprocess()
```

```python
@app.post("/embed_text/", response_model=Text, response_class=DocumentResponse)
async def embed_text(doc: Text) -> Text:
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        with torch.inference_mode():
            text_preprocess(doc)
            da = DocList[Text]([doc], tensor_type=TorchTensor).to_doc_vec()
            da.to(DEVICE)
            doc.embedding = text_encoder(da)[0].to('cpu')
    return doc
```

You can see that our earlier definition of the `Text` document now doubles as the API schema for the `/embed_text` endpoint.

With this running, we can query our model over the network:

```python
from httpx import AsyncClient
```

```python
text_input = Text(text='a picture of a rainbow')
```

```python
async with AsyncClient(
    app=app,
    base_url="http://test",
) as ac:
    response = await ac.post("/embed_text/", data=text_input.json())
```

```python
doc_resp = Text.parse_raw(response.content.decode())
```

```python
doc_resp.embedding.shape
```

And we're done! You have trained and served a multimodal ML model, with zero headaches and a lot of DocArray!
