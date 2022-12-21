(embed-via-model)=
# Embed via Neural Network

When DocumentArray `.tensors` being set,
you can use a neural network to {meth}`~docarray.array.mixins.embed.EmbedMixin.embed` it into it's vector representations,
i.e. filling `.embeddings`.

Embeddings can be used to measure the relatedness of your data.
Common cases include neural search, recommendation, clustering, outlier detection, deduplication etc.

Docarray provides an easy interface which allows you to encode your data with
the {meth}`~docarray.array.mixins.embed.EmbedMixin.embed` method.

After calling {meth}`~docarray.array.mixins.embed.EmbedMixin.embed`,
you should expect vector representation of your data has been stored in `.embeddings` endpoint.

Docarray {meth}`~docarray.array.mixins.embed.EmbedMixin.embed` endpoint support:

1. Mainstream deep learning frameworks, including Pytorch, Tensorflow, ONNX and PaddlePaddle.
2. Both CPU and CUDA devices.
3. Embed in batches.

### Embed for Computer Vision

````{tab} Torchvision ResNet50
```python
import numpy as np
from docarray import DocumentArray
from torchvision.models.resnet import resnet18

docs = DocumentArray.empty(10)
docs.tensors = np.random.rand(10, 3, 224, 224).astype('float32')

embed_model = resnet18()
docs.embed(embed_model=embed_model, device='cuda', device_id=0, batch_size=5)
```
````
````{tab} Tensorflow ResNet50
```python
import numpy as np
import tensorflow as tf
from docarray import DocumentArray

docs = DocumentArray.empty(10)
docs.tensors = np.random.rand(10, 224, 224, 3).astype('float32')

embed_model = tf.keras.applications.resnet50.ResNet50()
docs.embed(embed_model=embed_model, device='cuda', device_id=0, batch_size=5)
```
````
````{tab} ONNX ResNet50
```python
import numpy as np
import onnxruntime as ort
from docarray import DocumentArray

docs = DocumentArray.empty(10)
docs.tensors = np.random.rand(10, 3, 224, 224).astype('float32')
# please download model here: https://github.com/onnx/models/blob/main/vision/classification/resnet/model/resnet50-v1-12.onnx
ort_session = ort.InferenceSession('~your-path/resnet50-v1-12.onnx')
docs.embed(embed_model=ort_session, device='cuda', device_id=0, batch_size=5)
```
````
````{tab} PaddlePaddle ResNet50
```python
import numpy as np
import paddle
from docarray import DocumentArray

docs = DocumentArray.empty(10)
docs.tensors = np.random.rand(10, 3, 224, 224).astype('float32')

embed_model = paddle.vision.models.resnet50()
docs.embed(embed_model=embed_model, device='cuda', device_id=0, batch_size=5)
```
````
````{tab} Timm ResNet50
```python
import numpy as np
import timm
from docarray import DocumentArray

docs = DocumentArray.empty(10)
docs.tensors = np.random.rand(10, 3, 224, 224).astype('float32')

embed_model = timm.create_model('resnet50')
docs.embed(embed_model=embed_model, device='cuda', device_id=0, batch_size=5)
```
````

```{important}
In practice, when using a pre-trained network such as ResNet,
please remove the last fully-connected layer.
This is because the last layers are used to optimize objectives such as class scores,
not suitable for serving as embedding layers.
```


### Embed with Transformers

````{tab} Huggingface Transformers
```python
from docarray import DocumentArray
from transformers import BertModel, BertTokenizer

model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def collate_fn(da):
    """Tokenize `texts` field of the DocumentArray"""
    return tokenizer(
        da.texts,
        return_tensors='pt',
    )

docs = DocumentArray.empty(1)
docs.texts = ['embed me!']
docs.embed(model, collate_fn=collate_fn)
```
````
````{tab} Sentence Transformers
```python
from docarray import DocumentArray
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

docs = DocumentArray.empty(1)
docs.texts = ['embed me!']
docs.embeddings = model.encode(docs.texts)
```
````

### Embed with Cohere & OpenAI API

````{tab} Cohere
```python
import cohere
import numpy as np
from docarray import DocumentArray

token = ''
co = cohere.Client(token)

docs = DocumentArray.empty(1)
docs.texts = ['embed me!']
docs.embeddings = np.array(
    co.embed(docs.texts).embeddings
)
```
````
````{tab} OpenAI
```python
import openai
import numpy as np
from docarray import DocumentArray

openai.api_key = ''

docs = DocumentArray.empty(1)
docs.texts = ['embed me!']

for doc in docs:
    doc.embedding = np.array(
        openai.Embedding.create(input=doc.text, engine="text-embedding-ada-002")[
            "data"
        ][0]['embedding']
    )

```
````
