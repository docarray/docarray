# Tensor



DocArray support several tensor types that can be used inside BaseDoc. 

The main ones are:

- [`NdArray`][docarray.typing.tensor.NdArray] for numpy tensors
- [`TorchTensor`][docarray.typing.tensor.TorchTensor] for Pytorch tensors
- [`TensorFlowTensor`][docarray.typing.tensor.TensorFlowTensor] for tensorflow tensors

The three of them wrap from their respective framework tensor type. 

!!! note
    [`NdArray`][docarray.typing.tensor.NdArray] and [`TorchTensor`][docarray.typing.tensor.TorchTensor] are a subclass of their native tensor type. This means that they can be used natively inside their framework.

!!! warning
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

## Type coercion with different tensor types 


DocArray also supports type coercion between different tensor types. This mean that if you pass a different tensor type to a tensor field, it will be converted to the right tensor type.

For instance if you define a field of type [`TorchTensor`][docarray.typing.tensor.TorchTensor] and you pass a numpy array to it, it will be converted to a [`TorchTensor`][docarray.typing.tensor.TorchTensor].

```python
from docarray import BaseDoc
from docarray.typing import TorchTensor
import numpy as np


class MyTensorsDoc(BaseDoc):
    tensor: TorchTensor


doc = MyTensorsDoc(tensor=np.zeros(512))
doc.summary()
```

```bash
📄 MyTensorsDoc : 0a10f88 ...
╭─────────────────────┬────────────────────────────────────────────────────────╮
│ Attribute           │ Value                                                  │
├─────────────────────┼────────────────────────────────────────────────────────┤
│ tensor: TorchTensor │ TorchTensor of shape (512,), dtype: torch.float64      │
╰─────────────────────┴────────────────────────────────────────────────────────╯
```

And the same will work in the other direction

```python
from docarray import BaseDoc
from docarray.typing import NdArray
import torch


class MyTensorsDoc(BaseDoc):
    tensor: NdArray


doc = MyTensorsDoc(tensor=torch.zeros(512))
doc.summary()
```

```bash
📄 MyTensorsDoc : 157e6f5 ...
╭─────────────────┬────────────────────────────────────────────────────────────╮
│ Attribute       │ Value                                                      │
├─────────────────┼────────────────────────────────────────────────────────────┤
│ tensor: NdArray │ NdArray of shape (512,), dtype: float32                    │
╰─────────────────┴────────────────────────────────────────────────────────────╯
```

!!! warning
    Type coercion from [`TensorFlowTensor`][docarray.typing.tensor.TensorFlowTensor] to [`TorchTensor`][docarray.typing.tensor.TorchTensor] and vice versa is not supported yet.


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
    If you don't precise the `tensor_type` parameter and you tensor field is a Union of tensor or [`AnyTensor`][docarray.typing.tensor.AnyTensor] it will use NdArray as default.
