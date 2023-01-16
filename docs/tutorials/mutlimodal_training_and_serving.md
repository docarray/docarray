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

# Multi-Modal Deep learning with DocArray

DocArray is a library for representing, sending, and storing multi-modal data that can be used for a variety of different
use cases.

Here we will focus on a workflow familiar to many ML Engineers: Building and training a model, and then serving it to
users.

This notebook contains two parts:

1. **Representing**: We will use DocArray to represent multi-modal data while **building and training a PyTorch model**.
We will see how DocArray can help to organize and group your modalities and tensors and make clear what methods expect as inputs and return as outputs.
2. **Sending**: We will take the model that we built and trained in part 1, and **serve it using FastAPI**.
We will see how DocArray narrows the gap between model development and model deployment, and how the same data models can be
reused in both contexts. That part will be very short, but that's the point!

So without further ado, let's dive into it!

# 1. Representing: Build and train a PyTorch model

We will train a [CLIP](https://arxiv.org/abs/2103.00020)-like model on a dataset composes of text-image-pairs.
The goal is to obtain a model that is able to understand both text and images and project them into a common embedding space.

We train the CLIP-like model on the [flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) dataset.
To run this notebook you need to download and unzip the data into the same folder as the notebook.

Not that in this notebook by no means we aim at reproduce any CLIP results (our dataset is way too small anyways),
but rather we want to show how DocArray datastructures help researchers and practitioners to write beautiful and 
pythonic multi-modal PyTorch code.

```python tags=[]
#!pip install "git+https://github.com/docarray/docarray@feat-rewrite-v2#egg=docarray[torch,image]"
#!pip install torchvision
#!pip install transformers
#!pip install fastapi
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
DEVICE = "cuda:2"  # change to your favourite device
```

<!-- #region tags=[] -->
## Create the Documents for handling the Muti-Modal data
<!-- #endregion -->

The first thing we are trying to achieve when using DocArray is to clearly model our data so that we never get confused
about which tensors are supposed to represent what.

To do that we are using a concept that is at the core of DocArray. The `Document`, a collection of multi-modal data.
The `BaseDocument` class allows users to define their own (nested, multi-modal) Document schema to represent any kind of complex data.

Let's start by defining a few Documents to handle the different modalities that we will use during our training:

```python
from docarray import BaseDocument, DocumentArray
from docarray.typing import TorchTensor, ImageUrl
```

Let's first create a Document for our Text modality. It will contain a number of `Tokens`, which we also define:

```python
from docarray.documents import Text as BaseText


class Tokens(BaseDocument):
    input_ids: TorchTensor[512]
    attention_mask: TorchTensor
```

```python
class Text(BaseText):
    tokens: Optional[Tokens]
```
Notice the `TorchTensor` type. It is a thin wrapper around `torch.Tensor` that can be use like any other torch tensor, 
but also enables additional features. One such feature is shape parametrization (`TorchTensor[512]`), which lets you
hint and even enforce the desired shape of any tensor!

To represent our image data, we use the `Image` Document that is included in DocArray:

```python
from docarray.documents import Image
```

Under the hood, an `Image` looks something like this (with the only main difference that it can take tensors from any
supported ML framework):

```python
# class Image(BaseDocument):
#     url: Optional[ImageUrl]
#     tensor: Optional[TorchTesor]
#     embedding: Optional[TorchTensor]
```

Actually, the `BaseText` above also alredy includes `tensor`, `url` and `embedding` fields, so we can use those on our
`Text` Document as well.

The final Document used for training here is the `PairTextImage`, which simply combines the Text and Image modalities:

```python
class PairTextImage(BaseDocument):
    text: Text
    image: Image
```

## Create the Dataset 


In this section we will create a multi-modal pytorch dataset around the Flick8k dataset using DocArray.

We will use DocArray data loading functionality to load the data and use Torchvision and Transformers to preprocess the data before feeding it to our deep learning model:

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

    def __call__(self, url: ImageUrl) -> TorchTensor[3, 224, 224]:
        return self.transform(url.load())
```

```python
class TextPreprocess:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")

    def __call__(self, text: str) -> Tokens:
        return Tokens(**self.tokenizer(text, padding="max_length", truncation=True))
```

`VisionPreprocess` and `TextPreprocess` implement standard preprocessing steps for images and text, nothing special here.

```python
class PairDataset(Dataset):
    def __init__(
        self,
        file: str,
        vision_preprocess: VisionPreprocess,
        text_preprocess: TextPreprocess,
        N=None,
    ):
        self.docs = DocumentArray[PairTextImage]([])

        with open("captions.txt", "r") as f:
            lines = list(f.readlines())
            lines = lines[1:N] if N else lines[1:]
            for line in lines:
                line = line.split(",")
                doc = PairTextImage(
                    text=Text(text=line[1]), image=Image(url=f"Images/{line[0]}")
                )
                self.docs.append(doc)

        self.vision_preprocess = vision_preprocess
        self.text_preprocess = text_preprocess

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, item):
        doc = self.docs[item].copy()
        doc.image.tensor = self.vision_preprocess(doc.image.url)
        doc.text.tokens = self.text_preprocess(doc.text.text)
        return doc

    @staticmethod
    def collate_fn(batch: List[PairTextImage]):
        batch = DocumentArray[PairTextImage](batch, tensor_type=TorchTensor)
        batch = batch.stack()

        return batch
```

In the `PairDataset` class we can already see some of the beauty of DocArray.
The dataset will return Documents that contain the text and image data, accessible via `doc.text` and `doc.image`.

Now let's instantiate this dataset:

```python
vision_preprocess = VisionPreprocess()
text_preprocess = TextPreprocess()
```

```python
dataset = PairDataset("captions.txt", vision_preprocess, text_preprocess)
loader = DataLoader(
    dataset, batch_size=64, collate_fn=PairDataset.collate_fn, shuffle=True
)
```

## Create the Pytorch model that works on DocumentArray


In this section we create two encoders, one per modality (Text and Image). These encoders are normal PyTorch `nn.Module`s.
The only difference is that they operate on DocumentArray rather that on torch.Tensor:

```python
class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

    def forward(self, texts: DocumentArray[Text]) -> TorchTensor:
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

The `TextEncoder` takes a `DocumentArray` of `Text`s as input, and returns an embedding `TorchTensor` as output.
`DocumentArray` can be seen as a list of `Text` documents, and the encoder will treat it as one batch.


```python
class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True)
        self.linear = nn.LazyLinear(out_features=768)

    def forward(self, images: DocumentArray[Image]) -> TorchTensor:
        x = self.backbone(images.tensor)
        return self.linear(x)
```

Similarly, the `VisionEncoder` also takes a `DocumentArray` of `Image`s as input, and returns an embedding `TorchTensor` as output.
However, it operates on the `image` attribute of each Document.

Now we can instantiate our encoders:

```python
vision_encoder = VisionEncoder().to(DEVICE)
text_encoder = TextEncoder().to(DEVICE)
```

As you can see, DocArray helps us to clearly convey what data is expected as input and output for each method, all through Python type hints.

## Train the model in a contrastive way between Text and Image (CLIP)


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
def clip_loss(image: DocumentArray[Image], text: DocumentArray[Text]) -> TorchTensor:
    sims = cosine_sim(image.embedding, text.embedding)
    return torch.norm(sims - torch.eye(sims.shape[0], device=DEVICE))
```

In the type hints of `cosine_sim` and `clip_loss` you can again notice that we can treat a `TorchTensor` like any other
`torch.Tensor`, and how we can make explicit what kind of data and data modalities the different functions expect.

```python
num_epoch = 1  # here you should do more epochs to really learn something
```

One things to notice here is that our dataloader does not return a `torch.Tensor` but a `DocumentArray[PairTextImage]`,
which is exactly what our model can operate on.

So let's write a training loop and train our encoders:

```python tags=[]
with torch.autocast(device_type="cuda", dtype=torch.float16):
    for epoch in range(num_epoch):
        for i, batch in enumerate(loader):  
            batch.to(DEVICE)  # DocumentArray can be moved to device

            optim.zero_grad()
            # FORWARD PASS:
            batch.image.embedding = vision_encoder(batch.image)
            batch.text.embedding = text_encoder(batch.text)
            loss = clip_loss(batch.image, batch.text)
            if i % 10 == 0:
                print(f"{i+epoch} steps , loss : {loss}")
            loss.backward()
            optim.step()
```

Here we can see how we can immediately group the output of each encoder with the Document (and modality) it belong to.

And with all that, we've successfully trained a CLIP-like model without ever being confused the meaning of any tensors!

# 1. Sending: Serve the model using FastAPI

Now that we have a trained CLIP model, let's see how we can serve this model with a REST API by reusing most of the code above.

Let's use our beloved [FastAPI](https://fastapi.tiangolo.com/) for that!


FastAPI is powerful because it allows you to define your Rest API data schema in pure Python.
And DocArray is fully compatible with FastAPI and Pydantic, which means that as long as you have a function that takes a Document as input, 
FastAPI will be able to automatically translate it into a fully fledged API with documentation, openAPI specification and more:

```python
from fastapi import FastAPI
from docarray.base_document import DocumentResponse
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
@app.post("/embed_text/", response_model=Text, response_class=DocumentResponse)
async def embed_text(doc: Text) -> Text:
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        with torch.inference_mode():
            doc.tokens = text_preprocess(doc.text)
            da = DocumentArray[Text]([doc], tensor_type=TorchTensor).stack()
            da.to(DEVICE)
            doc.embedding = text_encoder(da)[0].to('cpu')
    return doc
```

You can see that our earlier definition of the `Text` Document now doubles as the API schema for the `/embed_text` endpoint.

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

And we're done! You have trained and served a mulit-modal ML model, with zero headache and a lot of DocArray!
