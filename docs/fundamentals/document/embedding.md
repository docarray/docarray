# Embedding

Embedding is a multi-dimensional representation of a `Document` (often a `[1, D]` vector). It serves as a very important piece in the neural search. 

Document has an attribute {attr}`~docarray.Document.embedding` to contain the embedding information.

Like `.blob`, you can assign it with a Python (nested) List/Tuple, Numpy `ndarray`, SciPy sparse matrix (`spmatrix`), TensorFlow dense and sparse tensor, PyTorch dense and sparse tensor, or PaddlePaddle dense tensor.

```python
import numpy as np
import scipy.sparse as sp
import torch
import tensorflow as tf
from jina import Document

d0 = Document(embedding=[1, 2, 3])
d1 = Document(embedding=np.array([1, 2, 3]))
d2 = Document(embedding=np.array([[1, 2, 3], [4, 5, 6]]))
d3 = Document(embedding=sp.coo_matrix([0, 0, 0, 1, 0]))
d4 = Document(embedding=torch.tensor([1, 2, 3]))
d5 = Document(embedding=tf.sparse.from_dense(np.array([[1, 2, 3], [4, 5, 6]])))
```

## Fill embedding from DNN model

```{admonition} On multiple Documents
:class: tip

This is a syntax sugar on single Document, which leverages {meth}`~jina.types.arrays.mixins.embed.EmbedMixin.embed` underneath. To embed multiple Documents, do not use this feature in a for-loop. Instead, read more details in {ref}`embed-via-model`.    
```

Once a `Document` has `.blob` set, you can use a deep neural network to {meth}`~jina.types.arrays.mixins.embed.EmbedMixin.embed` it, which means filling `Document.embedding`. For example, our `Document` looks like the following:

```python
q = (Document(uri='/Users/hanxiao/Downloads/left/00003.jpg')
     .load_uri_to_image_blob()
     .set_image_blob_normalization()
     .set_image_blob_channel_axis(-1, 0))
```

Let's embed it into vector via ResNet:

```python
import torchvision
model = torchvision.models.resnet50(pretrained=True)
q.embed(model)
```

## Find nearest-neighbours

```{admonition} On multiple Documents
:class: tip

This is a syntax sugar on single Document, which leverages  {meth}`~jina.types.arrays.mixins.match.MatchMixin.match` underneath. To match multiple Documents, do not use this feature in a for-loop. Instead, find out more in {ref}`match-documentarray`.  
```

Once a Document has `.embedding` filled, it can be "matched". In this example, we build ten Documents and put them into a {ref}`DocumentArray<da-intro>`, and then use another Document to search against them.

```python
from jina import DocumentArray, Document
import numpy as np

da = DocumentArray.empty(10)
da.embeddings = np.random.random([10, 256])

q = Document(embedding=np.random.random([256]))
q.match(da)

print(q.matches[0])
```

```console
<jina.types.document.Document ('id', 'embedding', 'adjacency', 'scores') at 8256118608>
```



