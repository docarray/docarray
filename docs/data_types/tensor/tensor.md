# Tensor



DocArray support several tensor type that can be used inside BaseDoc. 

The main ones are:

- [`NdArray`][docarray.typing.tensor.NdArray] for numpy tensor
- [`TorchTensor`][docarray.typing.tensor.TorchTensor] for Pytorch tensor
- [`TensorFlowTensor`][docarray.typing.tensor.TensorFlowTensor] for tensorflow tensor

The three of them inherit from their respective framework tensor type. This means that they can be used natively inside their framework.

We also have a [`AnyTensor`][docarray.typing.tensor.AnyTensor] that is the Union of the three previous tensor type. 
This is a generic placeholder to specify that it can work with any tensor type (numpy, torch, tensorflow).

## Tensor typed validation

You don't need to instantiate directly the  [`NdArray`][docarray.typing.tensor.NdArray] , [`TorchTensor`][docarray.typing.tensor.TorchTensor], [`TensorFlowTensor`][docarray.typing.tensor.TensorFlowTensor] directly by yourself. They are only use as type hint field fot the [`BaseDoc`][docarray.base_doc.doc.BaseDoc] validation.
[`BaseDoc`][docarray.base_doc.doc.BaseDoc] will cast the native tensor type into the respective DocArray tensor type.

Let's take an example 

```python
from docarray import BaseDoc
from docarray.typing import NdArray

import numpy as np


class MyDoc(BaseDoc):
    tensor: NdArray


doc = MyDoc(tensor=np.zeros(100))

assert isinstance(doc.tensor, NdArray)  # True
``` 
here you see that the `doc.tensor` is a `NdArray`. 

```python
assert isinstance(doc.tensor, np.ndarray)  # True as well
``` 

But since it inherits from `np.ndarray` you can use it as a numpy array.

## Type coercion from numpy to pytorch 

!!! note
    If you pass a numpy array to a pytorch tensor field, the numpy array will be converted to a pytorch tensor. 


example:


```python
from docarray import BaseDoc
from docarray.typing import TorchTensor
import numpy as np


class MyTensorsDoc(BaseDoc):
    tensor1: TorchTensor[512]
    tensor2: TorchTensor[512]


rand_series_f64 = np.random.rand(512).astype('float64')
rand_series_f32 = np.random.rand(512).astype('float32')

doc = MyTensorsDoc(tensor1=rand_series_f64, tensor2=rand_series_f32)
doc.summary()
```

```
📄 MyTensorsDoc : 84877dd ...
╭──────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────╮
│ Attribute            │ Value                                                                                    │
├──────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤
│ tensor1: TorchTensor │ TorchTensor of shape (512,), dtype: torch.float64                                        │
│ tensor2: TorchTensor │ TorchTensor of shape (512,), dtype: torch.float32                                        │
╰──────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────╯
```


But this is not the case if you pass a numpy array to a tensorflow tensor field. 

!!! warning 
    Tensor field type coercion is only supported from numpy to pytorch, not the other way around.


## DocVec with AnyTensor

DocVec can be used with a BaseDoc which has a field of [`AnyTensor`][docarray.typing.tensor.AnyTensor] or a Union of tensor type. 

 But to do so DocVec needs to know the tensor type of the tensor field beforehand to create the right column.
 
You can precise these parameters with the `tensor_type` parameter of the [`DocVec`][docarray.vectorizer.doc_vec.DocVec] constructor.

```python
from docarray import BaseDoc, DocVec
from docarray.typing import AnyTensor, NdArray

import numpy as np


class MyDoc(BaseDoc):
    tensor: AnyTensor


docs = DocVec[MyDoc](
    [MyDoc(tensor=np.zeros(100)) for _ in range(10)], tensor_type=NdArray
)

assert isinstance(docs.tensor, NdArray)
```

!!! note
    If you don't precise the `tensor_type` parameter it will use NdArray as default.
