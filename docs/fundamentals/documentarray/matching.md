(match-documentarray)=
# Find Nearest Neighbours

```{important}

{meth}`~jina.types.arrays.mixins.match.MatchMixin.match` function supports both CPU & GPU, which can be specified by its `device` argument.
```

Once `embeddings` is set, one can use {func}`~jina.types.arrays.mixins.match.MatchMixin.match` function to find the nearest neighbour Documents from another `DocumentArray` based on their `.embeddings`.  

The following image visualizes how `DocumentArrayA` finds `limit=5` matches from the Documents in `DocumentArrayB`. By
default, the cosine similarity is used to evaluate the score between Documents.

```{figure} match_illustration_5.svg
:align: center
```

More generally, given two `DocumentArray` objects `da_1` and `da_2` the
function `da_1.match(da_2, metric=some_metric, normalization=(0, 1), limit=N)` finds for each Document in `da_1` the `N` Documents from `da_2` with the lowest metric values according to `some_metric`.

Note that, 

- `da_1.embeddings` and `da_2.embeddings` can be Numpy `ndarray`, Scipy sparse matrix, Tensorflow tensor, PyTorch tensor or Paddle tensor.
- `metric` can be `'cosine'`, `'euclidean'`,  `'sqeuclidean'` or a callable that takes two `ndarray` parameters and
  returns an `ndarray`.
- by default `.match` returns distance not similarity. One can use `normalization` to do min-max normalization. The min distance will be rescaled to `a`, the
  max distance will be rescaled to `b`; all other values will be rescaled into range `[a, b]`. For example, to convert the distance into [0, 1] score, one can use `.match(normalization=(1,0))`.
- `limit` represents the number of nearest neighbours.

The following example finds for each element in `da1` the three closest Documents from the elements in `da2` according to Euclidean distance.

````{tab} Dense embedding 
```{code-block} python
---
emphasize-lines: 20
---
import numpy as np
from jina import DocumentArray

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
from jina import DocumentArray

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

### Keep only ID

Default `A.match(B)` will copy the top-K matched Documents from B to `A.matches`. When these matches are big, copying them can be time-consuming. In this case, one can leverage `.match(..., only_id=True)` to keep only {attr}`~docarray.Document.id`:

```python
from jina import DocumentArray
import numpy as np

A = DocumentArray.empty(2)
A.texts = ['hello', 'world']
A.embeddings = np.random.random([2, 10])

B = DocumentArray.empty(3)
B.texts = ['long-doc1', 'long-doc2', 'long-doc3']
B.embeddings = np.random.random([3, 10])
```

````{tab} Only ID

```python
A.match(B, only_id=True)

for m in A.traverse_flat('m'):
    print(m.json())
```

```text
{
  "adjacency": 1,
  "id": "4a8ad5fe4f9b11ec90e61e008a366d48",
  "scores": {
    "cosine": {
      "value": 0.08097544
    }
  }
}
...
```

````

````{tab} Default (keep all attributes)

```python
A.match(B)

for m in A.traverse_flat('m'):
    print(m.json())
```

```text
{
  "adjacency": 1,
  "embedding": {
    "cls_name": "numpy",
    "dense": {
      "buffer": "csxkKGfE7T+/JUBkNzHiP3Lx96W4SdE/SVXrOxYv7T9Fmb+pp3rvP8YdsjGsXuw/CNbxUQ7v2j81AjCpbfjrP6g5iPB9hL4/PHljbxPi1D8=",
      "dtype": "<f8",
      "shape": [
        10
      ]
    }
  },
  "id": "9078d1ec4f9b11eca9141e008a366d48",
  "scores": {
    "cosine": {
      "value": 0.15957883
    }
  },
  "text": "long-doc1"
}
...
```

````

### GPU support

If `.embeddings` is a Tensorflow tensor, PyTorch tensor or Paddle tensor, `.match()` function can work directly on GPU. To do that, simply set `device=cuda`. For example,

```python
from jina import DocumentArray
import numpy as np
import torch

da1 = DocumentArray.empty(10)
da1.embeddings = torch.tensor(np.random.random([10, 256]))
da2 = DocumentArray.empty(10)
da2.embeddings = torch.tensor(np.random.random([10, 256]))

da1.match(da2, device='cuda')
```

````{tip}

When `DocumentArray`/`DocumentArrayMemmap` contain too many documents to fit into GPU memory, one can set `batch_size` to allievate the problem of OOM on GPU.

```python
da1.match(da2, device='cuda', batch_size=256)
```

````

Let's do a simple benchmark on CPU vs. GPU `.match()`:

```python
from jina import DocumentArray

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

Note that in the above GPU example we did a conversion. In practice, there is no need to do this conversion, `.embedding`/`.blob` as well as their bulk versions `.embeddings`/`.blobs` can store PyTorch/Tensorflow/Paddle/Scipy tensor **natively**. That is, in practice, you just need to assign the result directly into `.embeddings` in your Encoder via:

```python
da.embeddings = torch_model(da.blobs)  # <- no .numpy() is necessary
```

And then in just use `.match(da)`.

### Evaluate matches

You can easily evaluate the performance of matches via {func}`~jina.types.arrays.mixins.evaluation.EvaluationMixin.evaluate`, provide that you have the groundtruth of the matches.

Jina provides some common metrics used in the information retrieval community that allows one to evaluate the nearest-neighbour matches. These metrics include: precision, recall, R-precision, hit rate, NDCG, etc. The full list of functions can be found in {class}`~jina.math.evaluation`.

For example, let's create a `DocumentArray` with random embeddings and matching it to itself:

```python
import numpy as np
from jina import DocumentArray

da = DocumentArray.empty(10)
da.embeddings = np.random.random([10, 3])
da.match(da, exclude_self=True)
```

Now `da.matches` contains the matches. Let's use it as the groundtruth. Now let's create imperfect matches by mixing in ten "noise Documents" to every `d.matches`.

```python
da2 = copy.deepcopy(da)

for d in da2:
    d.matches.extend(DocumentArray.empty(10))
    d.matches = d.matches.shuffle()

print(da2.evaluate(da, metric='precision_at_k', k=5))
```

Now we should have the average Precision@10 close to 0.5.
```text
0.5399999999999999
```

Note that this value is an average number over all Documents of `da2`. If you want to look at the individual evaluation, you can check {attr}`~docarray.Document.evaluations` attribute, e.g.

```python
for d in da2:
    print(d.evaluations['precision_at_k'].value)
```

```text
0.4000000059604645
0.6000000238418579
0.5
0.5
0.5
0.4000000059604645
0.5
0.4000000059604645
0.5
0.30000001192092896
```

Note that `evaluate()` works only when two `DocumentArray` have the same length and their Documents are aligned by a hash function. The default hash function simply uses {attr}`~docarray.Document.id`. You can specify your own hash function.

(traverse-doc)=