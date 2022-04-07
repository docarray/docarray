# Process Modality

So far we have learned how to construct and select multimodal Document, we are now ready to leverage DocArray API/Jina/Hub Executor to process the modalities.

In a nutshell, you need to convert a multimodal dataclass to a Document object (or DocumentArray) before processing it. This is because DocArray API/Jina/Hub Executor always take Document/DocumentArray as the basic IO unit. The following figure illustrates the idea.

```{figure} img/process-mmdoc.svg

```


## Embed image and text via CLIP

Developed by OpenAI, CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It is also a perfect model to showcase multimodal dataclass processing.

Take the code snippet from [the original CLIP repository](https://github.com/openai/CLIP) as an example,

```python
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    print(image_features, text_features)
```

```text
tensor([[-7.3285e-02, -1.6554e-01, ..., -1.3394e-01, -5.5605e-01,  1.2397e-01]]) 

tensor([[ 0.0547, -0.0061,  0.0495,  ..., -0.6638, -0.1281, -0.4950],
        [ 0.1447,  0.0225, -0.2909,  ..., -0.4472, -0.3420,  0.1798],
        [ 0.1981, -0.2040, -0.1533,  ..., -0.4514, -0.5664,  0.0596]])
```

Let's refactor it via dataclass.

```{code-block} python
---
emphasize-lines: 4,5,10-22
---
import clip
import torch

from docarray import dataclass, DocumentArray, Document
from docarray.typing import Image, Text

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

@dataclass
class MMDoc:
    title: Text
    banner: Image = None

m1 = MMDoc(banner='CLIP.png', title='a diagram')
m2 = MMDoc(banner='CLIP.png', title='a dog')
m3 = MMDoc(banner='CLIP.png', title='a cat')

da = DocumentArray([Document(m1), Document(m2), Document(m3)])

image = preprocess(torch.tensor(da['@.[banner]'].tensors)).to(device)
text = clip.tokenize(da['@.[title]'].texts).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    print(image_features, text_features)
```

```text
tensor([[-7.3285e-02, -1.6554e-01, ..., -1.3394e-01, -5.5605e-01],
        [-7.3285e-02, -1.6554e-01, ..., -1.3394e-01, -5.5605e-01],
        [-7.3285e-02, -1.6554e-01, ..., -1.3394e-01, -5.5605e-01]]) 

tensor([[ 0.0547, -0.0061,  0.0495,  ..., -0.6638, -0.1281, -0.4950],
        [ 0.1447,  0.0225, -0.2909,  ..., -0.4472, -0.3420,  0.1798],
        [ 0.1981, -0.2040, -0.1533,  ..., -0.4514, -0.5664,  0.0596]])
```

## Embed via CLIP-as-service

[CLIP-as-service](https://github.com/jina-ai/clip-as-service) is a low-latency high-scalability service for embedding images and text. It can be easily integrated as a microservice into neural search solutions.

To use CLIP-as-service to process a dataclass object is extremely simple, which should also show you the idea to use existing Executors or services without touching their codebase.

1. Construct the dataclass.
    ```python
    from docarray import dataclass, field, Document, DocumentArray
    from docarray.typing import Text, Image


    @dataclass
    class MMDoc:
        title: Text
        banner: Image = field(setter=lambda v: Document(uri=v), getter=lambda d: d.uri)
    ```

2. Create multimodal dataclass objects:

    ```python
    m1 = MMDoc(banner='CLIP.png', title='a diagram')
    m2 = MMDoc(banner='CLIP.png', title='a dog')
    m3 = MMDoc(banner='CLIP.png', title='a cat')
    ```

3. Convert them into a DocumentArray.

   ```python
   da = DocumentArray([Document(m1), Document(m2), Document(m3)])
   ```

4. Select the modality via the selector syntax and send via client

    ```python
    from clip_client import Client

    c = Client('grpc://demo-cas.jina.ai:51000')
    print(c.encode(da['@.[banner]']).embeddings)
    print(c.encode(da['@.[title]']).embeddings)
    ```

```text
[[ 0.3137  -0.1458   0.303   ...  0.8877  -0.2546  -0.11365]
 [ 0.3137  -0.1458   0.303   ...  0.8877  -0.2546  -0.11365]
 [ 0.3137  -0.1458   0.303   ...  0.8877  -0.2546  -0.11365]]

[[ 0.05466  -0.005997  0.0498   ... -0.663    -0.1274   -0.4944  ]
 [ 0.1442    0.02275  -0.291    ... -0.4468   -0.3416    0.1798  ]
 [ 0.1985   -0.204    -0.1534   ... -0.4507   -0.5664    0.0598  ]]
```



