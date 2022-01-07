# Embedding

Embedding is a multi-dimensional representation of a Document (often a `[1, D]` vector). It serves as a very important piece in machine learning. The attribute {attr}`~docarray.Document.embedding` is designed to contain the embedding information of a Document.

Like `.blob`, you can assign it with a Python (nested) List/Tuple, Numpy `ndarray`, SciPy sparse matrix (`spmatrix`), TensorFlow dense and sparse tensor, PyTorch dense and sparse tensor, or PaddlePaddle dense tensor.

```python
import numpy as np
import scipy.sparse as sp
import torch
import tensorflow as tf

from docarray import Document

d0 = Document(embedding=[1, 2, 3])
d1 = Document(embedding=np.array([1, 2, 3]))
d2 = Document(embedding=np.array([[1, 2, 3], [4, 5, 6]]))
d3 = Document(embedding=sp.coo_matrix([0, 0, 0, 1, 0]))
d4 = Document(embedding=torch.tensor([1, 2, 3]))
d5 = Document(embedding=tf.sparse.from_dense(np.array([[1, 2, 3], [4, 5, 6]])))
```

Unlike some other packages, DocArray will not actively cast `dtype` into `float32`. If the right-hand assigment `dtype` is `float64` in PyTorch, it will stay as a PyTorch `float64` tensor.

To assign multiple Documents `.blob` and `.embedding` in bulk, you {ref}`should use DocumentArray<da-content-embedding>`. It is much faster and smarter than using for-loop.


## Fill embedding via neural network

```{admonition} On multiple Documents use DocumentArray
:class: tip

To embed multiple Documents, do not use this feature in a for-loop. Instead, put all Documents in a DocumentArray and call `.embed()`. You can find out more in {ref}`embed-via-model`.
```

Usually you don't want to assign embedding manually, but instead doing something like:

```text
d.blob   \
d.text   ---> some DNN model ---> d.embedding
d.buffer /
```

Once a Document has content field set, you can use a deep neural network to {meth}`~docarray.document.mixins.sugar.SingletonSugarMixin.embed` it, which means filling `.embedding`. For example, our Document looks like the following:

```python
q = (Document(uri='/Users/hanxiao/Downloads/left/00003.jpg')
     .load_uri_to_image_blob()
     .set_image_blob_normalization()
     .set_image_blob_channel_axis(-1, 0))
```

Let's embed it into vector via ResNet50:

```python
import torchvision
model = torchvision.models.resnet50(pretrained=True)
q.embed(model)
```

## Find nearest-neighbours

```{admonition} On multiple Documents use DocumentArray
:class: tip

To match multiple Documents, do not use this feature in a for-loop. Instead, find out more in {ref}`match-documentarray`.  
```

Documents have `.embedding` set can be "matched" against each other. In this example, we build ten Documents and put them into a {ref}`DocumentArray<da-intro>`, and then use another Document to search against them.

```python
from docarray import DocumentArray, Document
import numpy as np

da = DocumentArray.empty(10)
da.embeddings = np.random.random([10, 256])

q = Document(embedding=np.random.random([256]))
q.match(da)

q.summary()
```

```text
 <Document ('id', 'embedding', 'matches') at 63a39fa86d6911eca6fa1e008a366d49>
    └─ matches
          ├─ <Document ('id', 'adjacency', 'embedding', 'scores') at 63a39aee6d6911eca6fa1e008a366d49>
          ├─ <Document ('id', 'adjacency', 'embedding', 'scores') at 63a399d66d6911eca6fa1e008a366d49>
          ├─ <Document ('id', 'adjacency', 'embedding', 'scores') at 63a39b346d6911eca6fa1e008a366d49>
          ├─ <Document ('id', 'adjacency', 'embedding', 'scores') at 63a3999a6d6911eca6fa1e008a366d49>
          ├─ <Document ('id', 'adjacency', 'embedding', 'scores') at 63a39a626d6911eca6fa1e008a366d49>
          ├─ <Document ('id', 'adjacency', 'embedding', 'scores') at 63a397ba6d6911eca6fa1e008a366d49>
          ├─ <Document ('id', 'adjacency', 'embedding', 'scores') at 63a39a1c6d6911eca6fa1e008a366d49>
          ├─ <Document ('id', 'adjacency', 'embedding', 'scores') at 63a39ab26d6911eca6fa1e008a366d49>
          ├─ <Document ('id', 'adjacency', 'embedding', 'scores') at 63a399046d6911eca6fa1e008a366d49>
          └─ <Document ('id', 'adjacency', 'embedding', 'scores') at 63a399546d6911eca6fa1e008a366d49>
```



