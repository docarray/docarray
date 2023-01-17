(match-documentarray)=
# Find Nearest Neighbors

```{important}

{meth}`~docarray.array.mixins.match.MatchMixin.match` and {meth}`~docarray.array.mixins.find.FindMixin.find` support both CPU & GPU.
```

Once `.embeddings` is set, you can use the {meth}`~docarray.array.mixins.find.FindMixin.find` or {func}`~docarray.array.mixins.match.MatchMixin.match` method to find the nearest-neighbor Documents from another DocumentArray (or the current DocumentArray itself) based on their `.embeddings` and distance metrics.  

## Difference between find and match

Though both `.find()` and `.match()` are about finding nearest neighbors of a given "query" and both accept similar arguments, there are some differences:

##### Which side is the query on?

- `.find()` always requires the query on the right-hand side. Say you have a DocumentArray with one million Documents, to find a query's nearest neighbors you should use `one_million_docs.find(query)`;  
- `.match()` assumes the query is on left-hand side. `A.match(B)` semantically means "A matches against B and saves the results to A". So with `.match()` you should use `query.match(one_million_docs)`.

##### What's the query type?

- The query (on the right) in `.find()` can be a plain ndarray-like object, single Document, or DocumentArray.
- The query (on the left) in `.match()` can be either a Document or DocumentArray. 

##### What is the return?

- `.find()` returns a List of DocumentArrays, each of which corresponds to one element/row in the query.
- `.match()` doesn't return anything. Matched results are stored inside the left-hand side's `.matches`.

Moving forwards, we'll use `.match()`. But bear in mind you could also use `.find()` by switching the right and left-hand sides.

### Example

In the following example, for each element in `da1`, we'll find the three closest Documents from the elements in `da2` based on Euclidean distance.

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

The above example when using `.find()`:

```python
da2.find(da1, metric='euclidean', limit=3)
```

Or simply:

```python
da2.find(
    np.array([[0, 0, 0, 0, 1], [1, 0, 0, 0, 0], [1, 1, 1, 1, 0], [1, 2, 2, 1, 0]]),
    metric='euclidean',
    limit=3,
)
```

The following metrics are supported:

| Metric                                                                                                               | Frameworks                                |
|----------------------------------------------------------------------------------------------------------------------|-------------------------------------------|
| `cosine`                                                                                                             | SciPy, NumPy, TensorFlow, PyTorch, Paddle |
| `sqeuclidean`                                                                                                        | SciPy, NumPy, TensorFlow, PyTorch, Paddle |
| `euclidean`                                                                                                          | SciPy, NumPy, TensorFlow, PyTorch, Paddle |
| [Metrics supported by SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html) | SciPy                                     |
| User defined callable                                                                                                | Depending on the callable                 |

Note that the framework is chosen automatically based on the type of `.embeddings`. For example, if `.embeddings` is a TensorFlow Tensor, then TensorFlow is used for computing. One exception is when `.embeddings` is a NumPy `ndarray`, you can choose to compute with either NumPy or SciPy (by specifying `.match(..., use_scipy=True)`). 

By default `A.match(B)` copies the top-K matched Documents from B to `A.matches`. When these matches are big, copying can be time-consuming. In this case, you can leverage `.match(..., only_id=True)` to keep only {attr}`~docarray.Document.id`.

### GPU support

If `.embeddings` is a TensorFlow, PyTorch, or Paddle tensor, `.match()` can work directly on the GPU. To do this, set `device=cuda`:

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

Like {meth}`~docarray.array.mixins.embed.EmbedMixin.embed`, if a DocumentArray is too large to fit into GPU memory, you can set `batch_size` to alleviate the problem of OOM on GPU:

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

````{tab} on CPU via NumPy

```python
import numpy as np

da1.embeddings = np.random.random([Q, D]).astype(np.float32)
da2.embeddings = np.random.random([M, D]).astype(np.float32)
```

```python
da1.match(da2, only_id=True)
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
da1.match(da2, device='cuda', batch_size=1_000, only_id=True)
```

```text
3.97 s ± 6.35 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

````
