(embed-via-model)=
# Embed via Neural Network

When DocumentArray `.tensors` being set,
you can use a neural network to {meth}`~docarray.array.mixins.embed.EmbedMixin.embed` it into it's vector representations,
i.e. filling `.embeddings`.

Embeddings can be used to measure the relatedness of your data.
Embeddings are commonly used for *neural search*, *recommendation*, *clustering*, *outlier detection*, *deduplication* etc.

Docarray provides an easy interface which allows you to encode your data with
the {meth}`~docarray.array.mixins.embed.EmbedMixin.embed` method.

After calling {meth}`~docarray.array.mixins.embed.EmbedMixin.embed`,
you should expect vector representation of your data has been stored in `.embeddings` endpoint.

Docarray {meth}`~docarray.array.mixins.embed.EmbedMixin.embed` endpoint support:

1. Mainstream deep learning frameworks, including Pytorch, Tensorflow, ONNX and PaddlePaddle.
2. Both CPU and CUDA devices.
3. Embed in batches.

## Embed in action

### Embed for Computer Vision

````{tab} Torchvision ResNet50
```python
import numpy as np
from docarray import DocumentArray
from torchvision.models.resnet import resnet18

docs = DocumentArray.empty(10)
docs.tensors = np.random.rand([10, 3, 224, 224]).astype('float32')

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
docs.tensors = np.random.rand([10, 224, 224, 3]).astype('float32')

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
docs.tensors = np.random.rand([10, 3, 224, 224]).astype('float32')

ort_session = ort.InferenceSession('~your-path/resnet18-v1-7.onnx')
docs.embed(embed_model=ort_session, device='cuda', device_id=0, batch_size=5)
```
````
````{tab} PaddlePaddle ResNet50
```python
import numpy as np
import paddle
from docarray import DocumentArray

docs = DocumentArray.empty(10)
docs.tensors = np.random.rand([10, 3, 224, 224]).astype('float32')

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
docs.tensors = np.random.rand([10, 3, 224, 224]).astype('float32')

embed_model = timm.create_model('resnet50')
docs.embed(embed_model=embed_model, device='cuda', device_id=0, batch_size=5)
```
````


### Embed with Huggingface Transformers

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
docs.texts = ['this is some random text to embed']
docs.embed(model, collate_fn=collate_fn)
```

### Embed with Cohere & OpenAI API

````{tab} Cohere
```python
import cohere
import numpy as np
from docarray import DocumentArray

token = ''
co = cohere.Client(token)

docs = DocumentArray.empty(1)
docs.texts = ['this is some random text to embed']
docs.embeddings = np.array(
    co.embed(docs.texts).embeddings
)
```
````
````{tab} OpenAI
```python
import cohere
import numpy as np
from docarray import DocumentArray

token = ''
co = cohere.Client(token)

docs = DocumentArray.empty(1)
docs.texts = ['this is some random text to embed']
docs.embeddings = np.array(
    co.embed(docs.texts).embeddings
)
```
````
