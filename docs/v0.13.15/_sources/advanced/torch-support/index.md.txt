# PyTorch/Deep Learning Frameworks

DocArray can be easily integrated into PyTorch, Tensorflow, PaddlePaddle frameworks.

The `.embedding` and `.tensor` attributes in Document class can contain PyTorch sparse/dense tensor, Tensorflow sparse/dense tensor or PaddlePaddle dense tensor.

It means that if you store the Document on disk in `pickle` or `protobuf` with/o compression, or transit the Document over the network in `pickle` or `protobuf` with/o compression, the data type of `.embedding` and `.tensor` is preserved.

```python
import numpy as np
import paddle
import torch

from docarray import Document, DocumentArray

emb = np.random.random([10, 3])

da = DocumentArray(
    [
        Document(embedding=emb),
        Document(embedding=torch.tensor(emb).to_sparse()),
        Document(embedding=torch.tensor(emb)),
        Document(embedding=paddle.to_tensor(emb)),
    ]
)

da.save_binary('test.protobuf.gz')
```

Now let's load them again and check the data type:
```python
from docarray import DocumentArray

for d in DocumentArray.load_binary('test.protobuf.gz'):
    print(type(d.embedding))
```

```text
<class 'numpy.ndarray'>
<class 'torch.Tensor'>
<class 'torch.Tensor'>
<class 'paddle.Tensor'>
```

## Load, map, batch in one-shot

There is a very common pattern in the deep learning engineering: loading big data, mapping it via some function for preprocessing on GPU, and batching it to GPU for intensive deep learning stuff.

There are many pitfalls in this pattern when not implemented correctly, to name a few:
- data may not fit into memory;
- mapping via CPU only utilizes a single-core;
- data-draining problem: GPU is not fully utilized as data is blocked by the slow CPU preprocessing step.

DocArray provides a high-level function {meth}`~docarray.array.mixins.dataloader.DataLoaderMixin.dataloader` that allows you to do this in one-shot, avoiding all pitfalls. The following figure illustrates this function:

```{figure} dataloader.svg
:width: 80%
```

Say we have a one million 32 x 32 color images, which takes 3.14GB on the disk with `protocol='protobuf'` and `compress='gz'`. To process it: 

```python
import time

from docarray import DocumentArray


def cpu_job(da):
    time.sleep(2)
    print('cpu job done')
    return da


def gpu_job(da):
    time.sleep(1)
    print('gpu job done')


for da in DocumentArray.dataloader(
    'da.protobuf.gz', func=cpu_job, batch_size=64, num_worker=4
):
    gpu_job(da)
```

```text
cpu job done
cpu job done
cpu job done
cpu job done
GPU job done
cpu job done
cpu job done
GPU job done
cpu job done
cpu job done
GPU job done
cpu job done
GPU job done
cpu job donecpu job done
```

