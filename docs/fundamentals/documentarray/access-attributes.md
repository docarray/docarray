(bulk-access)=
# Bulk Access Contents

You can quickly access `.text`, `.blob`, `.buffer`, `.embedding` of all Documents in the DocumentArray without writing a for-loop.

`DocumentArray` provides the plural counterparts, i.e. {attr}`~jina.types.arrays.mixins.content.ContentPropertyMixin.texts`, {attr}`~jina.types.arrays.mixins.content.ContentPropertyMixin.buffers`, {attr}`~jina.types.arrays.mixins.content.ContentPropertyMixin.blobs`, {attr}`~jina.types.arrays.mixins.content.ContentPropertyMixin.embeddings` that allows you to **get** and **set** these properties in one shot. It is much more efficient than looping.

```python
from jina import DocumentArray

da = DocumentArray.empty(2)
da.texts = ['hello', 'world']

print(da[0], da[1])
```

```text
<jina.types.document.Document ('id', 'text') at 4520833232>
<jina.types.document.Document ('id', 'text') at 5763350672>
```

When accessing `.blobs` or `.embeddings`, it automatically ravels/unravels the ndarray (can be Numpy/TensorFlow/PyTorch/SciPy/PaddlePaddle) for you.

```python
import numpy as np
import scipy.sparse
from jina import DocumentArray

sp_embed = np.random.random([10, 256])
sp_embed[sp_embed > 0.1] = 0
sp_embed = scipy.sparse.coo_matrix(sp_embed) 

da = DocumentArray.empty(10)

da.embeddings = scipy.sparse.coo_matrix(sp_embed)

print('da.embeddings.shape=', da.embeddings.shape)

for d in da:
    print('d.embedding.shape=', d.embedding.shape)
```

```text
da.embeddings.shape= (10, 256)
d.embedding.shape= (1, 256)
d.embedding.shape= (1, 256)
d.embedding.shape= (1, 256)
d.embedding.shape= (1, 256)
d.embedding.shape= (1, 256)
d.embedding.shape= (1, 256)
d.embedding.shape= (1, 256)
d.embedding.shape= (1, 256)
d.embedding.shape= (1, 256)
d.embedding.shape= (1, 256)
```

### Bulk access to attributes

{meth}`~jina.types.arrays.mixins.getattr.GetAttributeMixin.get_attributes` let you fetch multiple attributes from the `Document`s in
one shot:

```{code-block} python
---
emphasize-lines: 9
---
import numpy as np

from jina import DocumentArray, Document

da = DocumentArray([Document(id=1, text='hello', embedding=np.array([1, 2, 3])),
                    Document(id=2, text='goodbye', embedding=np.array([4, 5, 6])),
                    Document(id=3, text='world', embedding=np.array([7, 8, 9]))])

da.get_attributes('id', 'text', 'embedding')
```

```text
[('1', '2', '3'), ('hello', 'goodbye', 'world'), (array([1, 2, 3]), array([4, 5, 6]), array([7, 8, 9]))]
```