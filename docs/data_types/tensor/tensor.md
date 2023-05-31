# 🔢 Tensor

DocArray supports several tensor types that can you can use inside `BaseDoc`. 

The main ones are:

- [`NdArray`][docarray.typing.tensor.NdArray] for NumPy tensors
- [`TorchTensor`][docarray.typing.tensor.TorchTensor] for PyTorch tensors
- [`TensorFlowTensor`][docarray.typing.tensor.TensorFlowTensor] for TensorFlow tensors

The three of them wrap their respective framework's tensor type. 

!!! note
    [`NdArray`][docarray.typing.tensor.NdArray] and [`TorchTensor`][docarray.typing.tensor.TorchTensor] are a subclass of their native tensor type. This means that they can be used natively in their framework.

!!! warning
    [`TensorFlowTensor`][docarray.typing.tensor.TensorFlowTensor] stores the pure `tf.Tensor` object inside the `tensor` attribute. This is due to a limitation of the TensorFlow framework that prevents you from subclassing the `tf.Tensor` object.

DocArray also supports [`AnyTensor`][docarray.typing.tensor.AnyTensor], which is the Union of the three previous tensor types. 
This is a generic placeholder to specify that it can work with any tensor type (NumPy, PyTorch, TensorFlow).

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

You can check that the number of axes is correct by specifying `NdArray['x','y']`, `TorchTensor['x','y']`, `TensorFlowTensor['x','y']`.

```python
from docarray import BaseDoc
from docarray.typing import NdArray


class MyDoc(BaseDoc):
    tensor: NdArray['x', 'y']
``` 

Here you can only pass a tensor with two axes. `np.zeros(10, 12)` will work, but `np.zeros(10, 12, 3)` will raise an error.

### Axis names

You can specify that two axes should have the same dimensions with the syntax `NdArray['x', 'x']`, `TorchTensor['x', 'x']`, `TensorFlowTensor['x', 'x']`.

```python
from docarray import BaseDoc
from docarray.typing import NdArray


class MyDoc(BaseDoc):
    tensor: NdArray['x', 'x']
``` 

Here you can only pass a tensor with two axes that have the same dimensions. `np.zeros(10, 10)` will work but `np.zeros(10, 12)` will raise an error.

### Arbitrary number of axis

To specify that your shape can have an arbitrary number of axes, use the syntax `NdArray['x', ...]`, or `NdArray[..., 'x']`.

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

Let's look at an example:

```python
from docarray import BaseDoc
from docarray.typing import NdArray

import numpy as np


class MyDoc(BaseDoc):
    tensor: NdArray


doc = MyDoc(tensor=np.zeros(100))

assert isinstance(doc.tensor, NdArray)  # True
``` 
Here you see that the `doc.tensor` is an `NdArray`:

```python
assert isinstance(doc.tensor, np.ndarray)  # True as well
``` 

But since it inherits from `np.ndarray`, you can also use it as a normal NumPy array. The same holds for PyTorch and `TorchTensor`.

## Type coercion with different tensor types 

DocArray also supports type coercion between different tensor types. This mean that if you pass a different tensor type to a tensor field, it will be converted to the correct tensor type.

For instance, if you define a field of type [`TorchTensor`][docarray.typing.tensor.TorchTensor] and you pass a NumPy array to it, it will be converted to a [`TorchTensor`][docarray.typing.tensor.TorchTensor].

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

It also works in the other direction:

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

## `DocVec` with `AnyTensor`

[`DocVec`][docarray.array.doc_vec.doc_vec.DocVec] can be used with a `BaseDoc` which has a field of [`AnyTensor`][docarray.typing.tensor.AnyTensor] or any other Union of tensor types. 

However, the `DocVec` needs to know the tensor type of the tensor field beforehand to create the correct column.
 
You can specify these parameters with the `tensor_type` parameter of the [`DocVec`][docarray.vectorizer.doc_vec.DocVec] constructor:

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
    `NdArray` will be used by default if:
    
    - you don't specify the `tensor_type` parameter
    - your tensor field is a Union of tensor or [`AnyTensor`][docarray.typing.tensor.AnyTensor]
