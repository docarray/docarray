# Tensor



DocArray support several tensor type that can be used to inside BaseDoc. 

The main one are : 
- [`NdArray`][docarray.typing.tensor.NdArray] for numpy tensor
- [`TorchTensor`][docarray.typing.tensor.TorchTensor] for Pytorch tensor
- [`TensorFlowTensor`][docarray.typing.tensor.TensorFlowTensor] for tensorflow tensor

The three of them inherit from their respective framework tensor type. This mean that they can be used natively inside their framework.

We also have a [`AnyTensor`][docarray.typing.tensor.AnyTensor] that is the Union of the three previous tensor type . 
This is a generic placeholder to specify that can work with any tensor type.

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

But since it inherit from `np.ndarray` you can use it as a numpy array.

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

