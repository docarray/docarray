(match-documentarray)=
# Match Nearest Neighbours

```{important}

{meth}`~docarray.array.mixins.match.MatchMixin.match` supports both CPU & GPU.
```

Once `.embeddings` is set, one can use {func}`~docarray.array.mixins.match.MatchMixin.match` function to find the nearest-neighbour Documents from another DocumentArray (or itself) based on their `.embeddings`.  

The following example finds for each element in `da1` the three closest Documents from the elements in `da2` according to Euclidean distance.

````{tab} Dense embedding 
```{code-block} python
---
emphasize-lines: 20
---
import numpy as np
from docarray import DocumentArray

da1 = DocumentArray.empty(4)
da1.embeddings = np.array(
    [[0, 0, 0, 0, 1], [1, 0, 0, 0, 0], [1, 1, 1, 1, 0], [1, 2, 2, 1, 0]]
)

da2 = DocumentArray.empty(5)
da2.embeddings = np.array(
    [
        [0.0, 0.1, 0.0, 0.0, 0.0],
        [1.0, 0.1, 0.0, 0.0, 0.0],
        [1.0, 1.2, 1.0, 1.0, 0.0],
        [1.0, 2.2, 2.0, 1.0, 0.0],
        [4.0, 5.2, 2.0, 1.0, 0.0],
    ]
)

da1.match(da2, metric='euclidean', limit=3)

query = da1[2]
print(f'query emb = {query.embedding}')
for m in query.matches:
    print('match emb =', m.embedding, 'score =', m.scores['euclidean'].value)
```

```text
query emb = [1 1 1 1 0]
match emb = [1.  1.2 1.  1.  0. ] score = 0.20000000298023224
match emb = [1.  2.2 2.  1.  0. ] score = 1.5620499849319458
match emb = [1.  0.1 0.  0.  0. ] score = 1.6763054132461548
```
````

````{tab} Sparse embedding


```{code-block} python
---
emphasize-lines: 21
---
import numpy as np
import scipy.sparse as sp
from docarray import DocumentArray

da1 = DocumentArray.empty(4)
da1.embeddings = sp.csr_matrix(np.array(
    [[0, 0, 0, 0, 1], [1, 0, 0, 0, 0], [1, 1, 1, 1, 0], [1, 2, 2, 1, 0]]
))

da2 = DocumentArray.empty(5)
da2.embeddings = sp.csr_matrix(np.array(
    [
        [0.0, 0.1, 0.0, 0.0, 0.0],
        [1.0, 0.1, 0.0, 0.0, 0.0],
        [1.0, 1.2, 1.0, 1.0, 0.0],
        [1.0, 2.2, 2.0, 1.0, 0.0],
        [4.0, 5.2, 2.0, 1.0, 0.0],
    ]
))

da1.match(da2, metric='euclidean', limit=3)

query = da1[2]
print(f'query emb = {query.embedding}')
for m in query.matches:
    print('match emb =', m.embedding, 'score =', m.scores['euclidean'].value)
```

```text
query emb =   (0, 0)	1
  (0, 1)	1
  (0, 2)	1
  (0, 3)	1
match emb =   (0, 0)	1.0
  (0, 1)	1.2
  (0, 2)	1.0
  (0, 3)	1.0 score = 0.20000000298023224
match emb =   (0, 0)	1.0
  (0, 1)	2.2
  (0, 2)	2.0
  (0, 3)	1.0 score = 1.5620499849319458
match emb =   (0, 0)	1.0
  (0, 1)	0.1 score = 1.6763054132461548
```

````

The following metrics are supported:

| Metric                                                                                                               | Frameworks                                |
|----------------------------------------------------------------------------------------------------------------------|-------------------------------------------|
| `cosine`                                                                                                             | Scipy, Numpy, Tensorflow, Pytorch, Paddle |
| `sqeuclidean`                                                                                                        | Scipy, Numpy, Tensorflow, Pytorch, Paddle |
| `euclidean`                                                                                                          | Scipy, Numpy, Tensorflow, Pytorch, Paddle |
| [Metrics supported by Scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html) | Scipy                                     |
| User defined callable                                                                                                | Depending on the callable                 |

Note that framework is auto-chosen based on the type of `.embeddings`. For example, if `.embeddings` is a Tensorflow Tensor, then Tensorflow will be used for computing. One exception is when `.embeddings` is a Numpy `ndarray`, you can choose to use Numpy or Scipy (by specify `.match(..., use_scipy=True)`) for computing. 

By default `A.match(B)` will copy the top-K matched Documents from B to `A.matches`. When these matches are big, copying them can be time-consuming. In this case, one can leverage `.match(..., only_id=True)` to keep only {attr}`~docarray.Document.id`.


### GPU support

If `.embeddings` is a Tensorflow tensor, PyTorch tensor or Paddle tensor, `.match()` function can work directly on GPU. To do that, simply set `device=cuda`. For example,

```python
from docarray import DocumentArray
import numpy as np
import torch

da1 = DocumentArray.empty(10)
da1.embeddings = torch.tensor(np.random.random([10, 256]))
da2 = DocumentArray.empty(10)
da2.embeddings = torch.tensor(np.random.random([10, 256]))

da1.match(da2, device='cuda')
```

Similar as in {meth}`~docarray.array.mixins.embed.EmbedMixin.embed`, if a DocumentArray is too large to fit into GPU memory, one can set `batch_size` to alleviate the problem of OOM on GPU.

```python
da1.match(da2, device='cuda', batch_size=256)
```

Let's do a simple benchmark on CPU vs. GPU `.match()`:

```python
from docarray import DocumentArray

Q = 10
M = 1_000_000
D = 768

da1 = DocumentArray.empty(Q)
da2 = DocumentArray.empty(M)
```

````{tab} on CPU via Numpy

```python
import numpy as np

da1.embeddings = np.random.random([Q, D]).astype(np.float32)
da2.embeddings = np.random.random([M, D]).astype(np.float32)
```

```python
%timeit da1.match(da2, only_id=True)
```

```text
6.18 s ± 7.18 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

````

````{tab} on GPU via PyTorch

```python
import torch

da1.embeddings = torch.tensor(np.random.random([Q, D]).astype(np.float32))
da2.embeddings = torch.tensor(np.random.random([M, D]).astype(np.float32))
```

```python
%timeit da1.match(da2, device='cuda', batch_size=1_000, only_id=True)
```

```text
3.97 s ± 6.35 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

````


