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

# MultiModal Deep learning with DocArray


 The goal of this notebook is to showcase the usage of `docarray` with 'pytorch' to do multi-modal machine learning.

We will train a [CLIP(https://arxiv.org/abs/2103.00020)-like model on a dataset compose of text and image. The goal is to train the model
that is able to understand both text and image and project them into a common embedding space.

We train the CLIP-like model on the [flick8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) dataset. To run this notebook you need to download and unzip the data into the same folder as the notebook.

In this notebook we don't aim at reproduce any CLIP results (our dataset is way to small anyway) but rather to show how DocArray datastructures help researcher to write  beautiful and pythonic multi-modal pytorch code

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
DEVICE = "cuda:2"
```

<!-- #region tags=[] -->
## Create the Documents for handling the MutiModal data
<!-- #endregion -->

At the heart of DocArray live the concept of `BaseDocument` that allow user to define a nested data schema to represent any kind of complex multi modal data. `BaseDocument` is a pythonic way to define a data schema, it is inspired by [Pydantic BaseModel](https://docs.pydantic.dev/usage/models/) (it is actually built on top of it)

Lets start to define Document to handle the different modalities that we will use during our training

```python
from docarray import BaseDocument, DocumentArray
from docarray.documents import Image
from docarray.documents import Text as BaseText
from docarray.typing import TorchTensor, ImageUrl
```

```python tags=[]
class Tokens(BaseDocument):
    input_ids: TorchTensor[512]
    attention_mask: TorchTensor
```

```python
class Text(BaseText):
    tokens: Optional[Tokens]
```

the final document use for training here is the PairTextImage which combine the Text and Image modalities

```python
class PairTextImage(BaseDocument):
    text: Text
    image: Image
```

## Create the Dataset 


In this section we will create a multi modal pytorch dataset around the Flick8k dataset using docarray.

We will use DocArray data loading functionality to load the data and use torchvision and transformers to preprocess the data before fedding it to our deepl learning model

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

## Create the Pytorch model that work on DocumentArray


In this section we create two encoders one for each modalities (Text and Image). These encoders are nornal pytorch nn.Module. the Only difference is that they operate on DocumentArray directly rather that on tensor

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

```python
vision_encoder = VisionEncoder().to(DEVICE)
text_encoder = TextEncoder().to(DEVICE)
```

## Train the model in a constrative way between Text and Image (CLIP)


Now that we have defined our dataloader and our models we can train the two encoder is a contrastive way. The goal is to match the representation of the text and the image for each pair in the dataset.

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

```python
num_epoch = 1  # here you should do more epochs to really learn something
```

One things to notice here is that our dataloader does not return a torch.Tensor but a DocumentArray[PairTextImage] ! 

```python tags=[]
with torch.autocast(device_type="cuda", dtype=torch.float16):
    for epoch in range(num_epoch):
        for i, batch in enumerate(loader):  
            batch.to(DEVICE)

            optim.zero_grad()
            batch.image.embedding = vision_encoder(batch.image)
            batch.text.embedding = text_encoder(batch.text)
            loss = clip_loss(batch.image, batch.text)
            if i % 10 == 0:
                print(f"{i+epoch} steps , loss : {loss}")
            loss.backward()
            optim.step()
```
## From prototype to production in, well, almost no line of code


Now we have a ML clip model trained ! Let's see how we can serve this model with a RestAPI by reusing most of the code above.

lets use our beloved [FastAPI](https://fastapi.tiangolo.com/)


FastAPI is powerfull because it allows you to define your RestAPI data schema only with python ! And DocArray is fully compatible with FastAPI that means that as long as you have a function that takes as input a Document FastAPI will be able to translate it into a fully fledge RestAPI with documentation, openAPI specification and more !

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

now we can test the API

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
