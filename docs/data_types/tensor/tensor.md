# Tensor



DocArray support several tensor types that can be used inside BaseDoc. 

The main ones are:

- [`NdArray`][docarray.typing.tensor.NdArray] for numpy tensors
- [`TorchTensor`][docarray.typing.tensor.TorchTensor] for Pytorch tensors
- [`TensorFlowTensor`][docarray.typing.tensor.TensorFlowTensor] for tensorflow tensors

The three of them wrap from their respective framework tensor type. 

!!! note
    [`NdArray`][docarray.typing.tensor.NdArray] and [`TorchTensor`][docarray.typing.tensor.TorchTensor] are a subclass of their native tensor type. This means that they can be used natively inside their framework.

!!! note
    [`TensorFlowTensor`][docarray.typing.tensor.TensorFlowTensor] stores the pure `tf.Tensor` object inside the `tensor` attribute. This is due to a limitation on the TensorFlow framework that does not allow to subclass the `tf.Tensor` object.

DocArray also supports [`AnyTensor`][docarray.typing.tensor.AnyTensor] which is the Union of the three previous tensor types. 
This is a generic placeholder to specify that it can work with any tensor type (numpy, torch, tensorflow).

## Tensor type validation

You don't need to directly instantiate the  [`NdArray`][docarray.typing.tensor.NdArray] , [`TorchTensor`][docarray.typing.tensor.TorchTensor], or [`TensorFlowTensor`][docarray.typing.tensor.TensorFlowTensor] by yourself.

Instead, you should use them as type hints on [`BaseDoc`][docarray.base_doc.doc.BaseDoc] fields, where they perform data validation.
During this process, [`BaseDoc`][docarray.base_doc.doc.BaseDoc] will cast the native tensor type into the respective DocArray tensor type.

Let's take an example:

```python
from docarray import BaseDoc
from docarray.typing import NdArray

import numpy as np


class MyDoc(BaseDoc):
    tensor: NdArray


doc = MyDoc(tensor=np.zeros(100))

assert isinstance(doc.tensor, NdArray)  # True
``` 
Here you see that the `doc.tensor` is an `NdArray`. 

```python
assert isinstance(doc.tensor, np.ndarray)  # True as well
``` 

But since it inherits from `np.ndarray` you can also use it as a normal numpy array. The same holds for pytorch and `TorchTensor`.

## Type coercion from NumPy to PyTorch 

!!! note
    If you pass a numpy array to a pytorch tensor field, the numpy array will be converted to a pytorch tensor. 


Example:


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

DocVec can be used with a BaseDoc which has a field of [`AnyTensor`][docarray.typing.tensor.AnyTensor] or any other Union of tensor types. 

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
