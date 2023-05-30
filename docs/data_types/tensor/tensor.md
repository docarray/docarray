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


## Tensor Shape validation

All three tensor types support shape validation. This means that you can specify the shape of the tensor using type hint syntax: `NdArray[100, 100]`, `TorchTensor[100, 100]`, `TensorFlowTensor[100, 100]`.

Let's take an example:

```python
from docarray import BaseDoc
from docarray.typing import NdArray


class MyDoc(BaseDoc):
    tensor: NdArray[100, 100]
``` 

If you try to pass a tensor with a different shape, an error will be raised:

```python
import numpy as np

try:
    doc = MyDoc(tensor=np.zeros((100, 200)))
except ValueError as e:
    print(e)
```

```bash
1 validation error for MyDoc
tensor
  cannot reshape array of size 20000 into shape (100,100) (type=value_error)
``` 


Whereas if you just pass a tensor with the correct shape, no error will be raised:

```python
doc = MyDoc(tensor=np.zeros((100, 100)))
``` 

### Axes validation

You can check that the number of axis is correct as well `NdArray['x','y']`, `TorchTensor['x','y']`, `TensorFlowTensor['x','y']`.

```python
from docarray import BaseDoc
from docarray.typing import NdArray


class MyDoc(BaseDoc):
    tensor: NdArray['x', 'y']
``` 

Here you can only pass a tensor with two axis. `np.zeros(10, 12)` will work but `np.zeros(10, 12, 3)` will raise an error.

### Axis names

You can as well specify that two axis should have the same dimension with the syntax `NdArray['x', 'x']`, `TorchTensor['x', 'x']`, `TensorFlowTensor['x', 'x']`.

```python
from docarray import BaseDoc
from docarray.typing import NdArray


class MyDoc(BaseDoc):
    tensor: NdArray['x', 'x']
``` 

Here you can only pass a tensor with two axis with the same dimension. `np.zeros(10, 10)` will work but `np.zeros(10, 12)` will raise an error.

### Arbitrary number of axis

You can specify that your shape can have an arbitrary number of axis with the syntax `NdArray['x', ...]`, or `NdArray[..., 'x']`

```python
from docarray import BaseDoc
from docarray.typing import NdArray


class MyDoc(BaseDoc):
    tensor: NdArray[100, ...]
``` 

Here you can only pass a tensor with at least one axis with dimension 100. `np.zeros(100, 10)` will work but `np.zeros(10, 12)` will raise an error.

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

[`DocVec`][docarray.array.doc_vec.doc_vec.DocVec] can be used with a BaseDoc which has a field of [`AnyTensor`][docarray.typing.tensor.AnyTensor] or any other Union of tensor types. 

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
