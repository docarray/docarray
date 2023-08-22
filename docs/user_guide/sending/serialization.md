# Serialization

DocArray offers various serialization options for all of its main data classes:
[BaseDoc][docarray.base_doc.doc.BaseDoc], [DocList][docarray.array.doc_list.doc_list.DocList], and [DocVec][docarray.array.doc_list.doc_list.DocVec]

## BaseDoc

You need to serialize a [BaseDoc][docarray.base_doc.doc.BaseDoc] before you can store or send it.

!!! note
    [BaseDoc][docarray.base_doc.doc.BaseDoc] supports serialization to `protobuf` and `json` formats.

### JSON

- [`json`][docarray.base_doc.doc.BaseDoc.json] serializes a [`BaseDoc`][docarray.base_doc.doc.BaseDoc] to a JSON string.
- [`parse_raw`][docarray.base_doc.doc.BaseDoc.parse_raw] deserializes a [`BaseDoc`][docarray.base_doc.doc.BaseDoc] from a JSON string.

```python
from typing import List
from docarray import BaseDoc


class MyDoc(BaseDoc):
    text: str
    tags: List[str]


doc = MyDoc(text='hello world', tags=['hello', 'world'])
json_str = doc.json()
new_doc = MyDoc.parse_raw(json_str)
assert doc == new_doc  # True
```

### Protobuf

- [`to_protobuf`][docarray.base_doc.mixins.io.IOMixin.to_protobuf] serializes a [`BaseDoc`][docarray.base_doc.doc.BaseDoc] to a `protobuf` message object.
- [`from_protobuf`][docarray.base_doc.mixins.io.IOMixin.from_protobuf] deserializes a [`BaseDoc`][docarray.base_doc.doc.BaseDoc] from a `protobuf` object.

```python
from typing import List
from docarray import BaseDoc


class MyDoc(BaseDoc):
    text: str
    tags: List[str]


doc = MyDoc(text='hello world', tags=['hello', 'world'])
proto_message = doc.to_protobuf()
new_doc = MyDoc.from_protobuf(proto_message)
assert doc == new_doc  # True
```

## DocList

When sending or storing [`DocList`][docarray.array.doc_list.doc_list.DocList], you need to use serialization.
[`DocList`][docarray.array.doc_list.doc_list.DocList] supports multiple ways to serialize the data.

### JSON

-  [`to_json()`][docarray.array.doc_list.io.IOMixinDocList.to_json] serializes a [`DocList`][docarray.array.doc_list.doc_list.DocList] to JSON. It returns the binary representation of the JSON object. 
-  [`from_json()`][docarray.array.doc_list.io.IOMixinDocList.from_json] deserializes a [`DocList`][docarray.array.doc_list.doc_list.DocList] from JSON. It can load from either a `str` or `binary` representation of the JSON object.

```python
from docarray import BaseDoc, DocList


class SimpleDoc(BaseDoc):
    text: str


dl = DocList[SimpleDoc]([SimpleDoc(text=f'doc {i}') for i in range(2)])

with open('simple-dl.json', 'wb') as f:
    json_dl = dl.to_json()
    print(json_dl)
    f.write(json_dl)

with open('simple-dl.json', 'r') as f:
    dl_load_from_json = DocList[SimpleDoc].from_json(f.read())
    print(dl_load_from_json)
```

```output
b'[{"id":"5540e72d407ae81abb2390e9249ed066","text":"doc 0"},{"id":"fbe9f80d2fa03571e899a2887af1ac1b","text":"doc 1"}]'
```

### Protobuf

- [`to_protobuf()`][docarray.array.doc_list.io.IOMixinDocList.to_protobuf] serializes a [`DocList`][docarray.array.doc_list.doc_list.DocList] to `protobuf`. It returns a `protobuf` object of `docarray_pb2.DocListProto` class.
- [`from_protobuf()`][docarray.array.doc_list.io.IOMixinDocList.from_protobuf] deserializes a [`DocList`][docarray.array.doc_list.doc_list.DocList] from `protobuf`. It accepts a `protobuf` message object to construct a [`DocList`][docarray.array.doc_list.doc_list.DocList].

```python
from docarray import BaseDoc, DocList


class SimpleDoc(BaseDoc):
    text: str


dl = DocList[SimpleDoc]([SimpleDoc(text=f'doc {i}') for i in range(2)])

proto_message_dl = dl.to_protobuf()
dl_from_proto = DocList[SimpleDoc].from_protobuf(proto_message_dl)
print(type(proto_message_dl))
print(dl_from_proto)
```

### Base64

When transferring data over the network, use `Base64` format to serialize the [`DocList`][docarray.array.doc_list.doc_list.DocList].
Serializing a [`DocList`][docarray.array.doc_list.doc_list.DocList] in Base64 supports both the `pickle` and `protobuf` protocols. You can also choose different compression methods.

- [`to_base64()`][docarray.array.doc_list.io.IOMixinDocList.to_base64] serializes a [`DocList`][docarray.array.doc_list.doc_list.DocList] to Base64
- [`from_base64()`][docarray.array.doc_list.io.IOMixinDocList.from_base64] deserializes a [`DocList`][docarray.array.doc_list.doc_list.DocList] from Base64:

You can multiple compression methods: `lz4`, `bz2`, `lzma`, `zlib`, and `gzip`.

```python
from docarray import BaseDoc, DocList


class SimpleDoc(BaseDoc):
    text: str


dl = DocList[SimpleDoc]([SimpleDoc(text=f'doc {i}') for i in range(2)])

base64_repr_dl = dl.to_base64(compress=None, protocol='pickle')

dl_from_base64 = DocList[SimpleDoc].from_base64(
    base64_repr_dl, compress=None, protocol='pickle'
)
```

### Save binary

These methods **serialize and save** your data:

- [`save_binary()`][docarray.array.doc_list.io.IOMixinDocList.save_binary] saves a [`DocList`][docarray.array.doc_list.doc_list.DocList] to a binary file.
- [`load_binary()`][docarray.array.doc_list.io.IOMixinDocList.load_binary] loads a [`DocList`][docarray.array.doc_list.doc_list.DocList] from a binary file.

You can choose between multiple compression methods: `lz4`, `bz2`, `lzma`, `zlib`, and `gzip`.

```python
from docarray import BaseDoc, DocList


class SimpleDoc(BaseDoc):
    text: str


dl = DocList[SimpleDoc]([SimpleDoc(text=f'doc {i}') for i in range(2)])

dl.save_binary('simple-dl.pickle', compress=None, protocol='pickle')

dl_from_binary = DocList[SimpleDoc].load_binary(
    'simple-dl.pickle', compress=None, protocol='pickle'
)
```

In the above snippet, the [`DocList`][docarray.array.doc_list.doc_list.DocList] is stored as the file `simple-dl.pickle`.

### Bytes

These methods just serialize your data, without saving it to a file:

- [to_bytes()][docarray.array.doc_list.io.IOMixinDocList.to_bytes] saves a [`DocList`][docarray.array.doc_list.doc_list.DocList] to a byte object.
- [from_bytes()][docarray.array.doc_list.io.IOMixinDocList.from_bytes] loads a [`DocList`][docarray.array.doc_list.doc_list.DocList] from a byte object.  

!!! note
    These methods are used under the hood by [save_binary()][docarray.array.doc_list.io.IOMixinDocList.to_base64] and [`load_binary()`][docarray.array.doc_list.io.IOMixinDocList.load_binary] to prepare/load/save to a binary file. You can also use them directly to work with byte files.

Like working with binary files:

- You can use `protocol` to choose between `pickle` and `protobuf`. 
- You can use multiple compression methods: `lz4`, `bz2`, `lzma`, `zlib`, and `gzip`.

```python
from docarray import BaseDoc, DocList


class SimpleDoc(BaseDoc):
    text: str


dl = DocList[SimpleDoc]([SimpleDoc(text=f'doc {i}') for i in range(2)])

bytes_dl = dl.to_bytes(protocol='pickle', compress=None)

dl_from_bytes = DocList[SimpleDoc].from_bytes(
    bytes_dl, compress=None, protocol='pickle'
)
```

### CSV

- [`to_csv()`][docarray.array.doc_list.io.IOMixinDocList.to_csv] serializes a [`DocList`][docarray.array.doc_list.doc_list.DocList] to a CSV file.
- [`from_csv()`][docarray.array.doc_list.io.IOMixinDocList.from_csv] deserializes a [`DocList`][docarray.array.doc_list.doc_list.DocList] from a CSV file.

Use the `dialect` parameter to choose the [dialect of the CSV format](https://docs.python.org/3/library/csv.html#dialects-and-formatting-parameters):

```python
from docarray import BaseDoc, DocList


class SimpleDoc(BaseDoc):
    text: str


dl = DocList[SimpleDoc]([SimpleDoc(text=f'doc {i}') for i in range(2)])

dl.to_csv('simple-dl.csv')
dl_from_csv = DocList[SimpleDoc].from_csv('simple-dl.csv')
print(dl_from_csv)
```

### Pandas.Dataframe

- [`from_dataframe()`][docarray.array.doc_list.io.IOMixinDocList.from_dataframe] loads a [`DocList`][docarray.array.doc_list.doc_list.DocList] from a [Pandas Dataframe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html).
- [`to_dataframe()`][docarray.array.doc_list.io.IOMixinDocList.to_dataframe] saves a [`DocList`][docarray.array.doc_list.doc_list.DocList] to a [Pandas Dataframe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html).

```python
from docarray import BaseDoc, DocList


class SimpleDoc(BaseDoc):
    text: str


dl = DocList[SimpleDoc]([SimpleDoc(text=f'doc {i}') for i in range(2)])

df = dl.to_dataframe()
dl_from_dataframe = DocList[SimpleDoc].from_dataframe(df)
print(dl_from_dataframe)
```

## DocVec

For sending or storing [`DocVec`][docarray.array.doc_vec.doc_vec.DocVec] it offers a very similar interface to that of
[`DocList`][docarray.array.doc_list.doc_list.DocList].

!!! note "Tensor type and (de)serialization"


    You can deserialize any serialized [DocVec][docarray.array.doc_list.doc_list.DocVec] to any tensor type ([`NdArray`][docarray.typing.tensor.NdArray], [`TorchTensor`][docarray.typing.tensor.TorchTensor], or [`TensorFlowTensor`][docarray.typing.tensor.TensorFlowTensor]),
    by passing the `tensor_type=...` parameter to the appropriate deserialization method.
    This is analogous to the `tensor_type=...` parameter in the [DocVec][docarray.array.doc_list.doc_list.DocVec.__init__] constructor.
    
    This means that you can choose at deserialization time if you are working with numpy, PyTorch, or TensorFlow tensors.
    
    If no `tensor_type` is passed, the default is `NdArray`.

### JSON

-  [`to_json()`][docarray.array.doc_vec.io.IOMixinDocVec.to_json] serializes a [`DocVec`][docarray.array.doc_vec.doc_vec.DocVec] to JSON. It returns the binary representation of the JSON object. 
-  [`from_json()`][docarray.array.doc_list.io.IOMixinDocVec.from_json] deserializes a [`DocList`][docarray.array.doc_vec.doc_vec.DocVec] from JSON. It can load from either a `str` or `binary` representation of the JSON object.

In contrast to [DocList's JSON format](#json-1), `DocVec.to_json()` outputs a column oriented JSON file:

```python
import torch
from docarray import BaseDoc, DocVec
from docarray.typing import TorchTensor


class SimpleDoc(BaseDoc):
    text: str
    tensor: TorchTensor


dv = DocVec[SimpleDoc](
    [SimpleDoc(text=f'doc {i}', tensor=torch.rand(64)) for i in range(2)]
)

with open('simple-dv.json', 'wb') as f:
    json_dv = dv.to_json()
    print(json_dv)
    f.write(json_dv)

with open('simple-dv.json', 'r') as f:
    dv_load_from_json = DocVec[SimpleDoc].from_json(f.read(), tensor_type=TorchTensor)
    print(dv_load_from_json)
```

```output
b'{"tensor_columns":{},"doc_columns":{},"docs_vec_columns":{},"any_columns":{"id":["005a208a0a9a368c16bf77913b710433","31d65f02cb94fc9756c57b0dbaac3a2c"],"text":["doc 0","doc 1"]}}'
<DocVec[SimpleDoc] (length=2)>
```

### Protobuf

- [`to_protobuf`][docarray.array.doc_vec.io.IOMixinDocVec.to_protobuf] serializes a [DocVec][docarray.array.doc_list.doc_list.DocVec] to `protobuf`. It returns a `protobuf` object of `docarray_pb2.DocVecProto` class. 
- [`from_protobuf`][docarray.array.doc_vec.io.IOMixinDocVec.from_protobuf] deserializes a [DocVec][docarray.array.doc_list.doc_list.DocVec] from `protobuf`. It accepts a protobuf message object to construct a [DocVec][docarray.array.doc_list.doc_list.DocVec].

```python
import numpy as np

from docarray import BaseDoc, DocVec
from docarray.typing import AnyTensor


class SimpleVecDoc(BaseDoc):
    tensor: AnyTensor


dv = DocVec[SimpleVecDoc]([SimpleVecDoc(tensor=np.ones(16)) for _ in range(8)])

proto_message_dv = dv.to_protobuf()

dv_from_proto = DocVec[SimpleVecDoc].from_protobuf(proto_message_dv)
```

You can deserialize any [DocVec][docarray.array.doc_list.doc_list.DocVec] protobuf message to any tensor type,
by passing the `tensor_type=...` parameter to [`from_protobuf`][docarray.array.doc_vec.io.IOMixinDocVec.from_protobuf]

This means that you can choose at deserialization time if you are working with numpy, PyTorch, or TensorFlow tensors.

If no `tensor_type` is passed, the default is `NdArray`.


```python
import torch

from docarray import BaseDoc, DocVec
from docarray.typing import TorchTensor, NdArray, AnyTensor


class AnyTensorDoc(BaseDoc):
    tensor: AnyTensor


dv = DocVec[AnyTensorDoc](
    [AnyTensorDoc(tensor=torch.ones(16)) for _ in range(8)], tensor_type=TorchTensor
)

proto_message_dv = dv.to_protobuf()

# deserialize to torch
dv_from_proto_torch = DocVec[AnyTensorDoc].from_protobuf(
    proto_message_dv, tensor_type=TorchTensor
)
assert dv_from_proto_torch.tensor_type == TorchTensor
assert isinstance(dv_from_proto_torch.tensor, TorchTensor)

# deserialize to numpy (default)
dv_from_proto_numpy = DocVec[AnyTensorDoc].from_protobuf(proto_message_dv)
assert dv_from_proto_numpy.tensor_type == NdArray
assert isinstance(dv_from_proto_numpy.tensor, NdArray)
```

!!! note
    Serialization to protobuf is not supported for union types involving `BaseDoc` types.

### Base64

When transferring data over the network, use `Base64` format to serialize the [DocVec][docarray.array.doc_list.doc_list.DocVec].
Serializing a [DocVec][docarray.array.doc_list.doc_list.DocVec] in Base64 supports both the `pickle` and `protobuf` protocols.
You can also choose different compression methods.

- [`to_base64()`][docarray.array.doc_vec.io.IOMixinDocVec.to_base64] serializes a [DocVec][docarray.array.doc_list.doc_list.DocVec] to Base64
- [`from_base64()`][docarray.array.doc_vec.io.IOMixinDocVec.from_base64] deserializes a [DocVec][docarray.array.doc_list.doc_list.DocVec] from Base64:

You can multiple compression methods: `lz4`, `bz2`, `lzma`, `zlib`, and `gzip`.

```python
from docarray import BaseDoc, DocVec
from docarray.typing import TorchTensor
import torch


class SimpleDoc(BaseDoc):
    text: str
    tensor: TorchTensor


dv = DocVec[SimpleDoc](
    [SimpleDoc(text=f'doc {i}', tensor=torch.rand(64)) for i in range(2)]
)

base64_repr_dv = dv.to_base64(compress=None, protocol='pickle')

dl_from_base64 = DocVec[SimpleDoc].from_base64(
    base64_repr_dv, compress=None, protocol='pickle', tensor_type=TorchTensor
)
```

### Save binary

These methods **serialize and save** your data:

- [`save_binary()`][docarray.array.doc_vec.io.IOMixinDocVec.save_binary] saves a [DocVec][docarray.array.doc_list.doc_list.DocVec] to a binary file.
- [`load_binary()`][docarray.array.doc_vec.io.IOMixinDocVec.load_binary] loads a [DocVec][docarray.array.doc_list.doc_list.DocVec] from a binary file.

You can choose between multiple compression methods: `lz4`, `bz2`, `lzma`, `zlib`, and `gzip`.

```python
from docarray import BaseDoc, DocVec
from docarray.typing import TorchTensor
import torch


class SimpleDoc(BaseDoc):
    text: str
    tensor: TorchTensor


dv = DocVec[SimpleDoc](
    [SimpleDoc(text=f'doc {i}', tensor=torch.rand(64)) for i in range(2)]
)

dv.save_binary('simple-dl.pickle', compress=None, protocol='pickle')

dv_from_binary = DocVec[SimpleDoc].load_binary(
    'simple-dv.pickle', compress=None, protocol='pickle', tensor_type=TorchTensor
)
```

In the above snippet, the [DocVec][docarray.array.doc_list.doc_list.DocVec] is stored as the file `simple-dv.pickle`.

### Bytes

These methods just serialize your data, without saving it to a file:

- [to_bytes()][docarray.array.doc_vec.io.IOMixinDocVec.to_bytes] saves a [DocVec][docarray.array.doc_list.doc_list.DocVec] to a byte object.
- [from_bytes()][docarray.array.doc_vec.io.IOMixinDocVec.from_bytes] loads a [DocVec][docarray.array.doc_list.doc_list.DocVec] from a byte object.  

!!! note
    These methods are used under the hood by [save_binary()][docarray.array.doc_vec.io.IOMixinDocVec.to_base64] and [`load_binary()`][docarray.array.doc_vec.io.IOMixinDocVec.load_binary] to prepare/load/save to a binary file. You can also use them directly to work with byte files.

Like working with binary files:

- You can use `protocol` to choose between `pickle` and `protobuf`. 
- You can use multiple compression methods: `lz4`, `bz2`, `lzma`, `zlib`, and `gzip`.

```python
from docarray import BaseDoc, DocVec
from docarray.typing import TorchTensor
import torch


class SimpleDoc(BaseDoc):
    text: str
    tensor: TorchTensor


dv = DocVec[SimpleDoc](
    [SimpleDoc(text=f'doc {i}', tensor=torch.rand(64)) for i in range(2)]
)

bytes_dv = dv.to_bytes(protocol='pickle', compress=None)

dv_from_bytes = DocVec[SimpleDoc].from_bytes(
    bytes_dv, compress=None, protocol='pickle', tensor_type=TorchTensor
)
```

### CSV

!!! warning
    [`DocVec`][docarray.array.doc_vec.doc_vec.DocVec] does not support `.to_csv()` or `from_csv()`.
    This is because CSV is a row-based format while DocVec has a column-based data layout.
    To overcome this, you can convert your [`DocVec`][docarray.array.doc_vec.doc_vec.DocVec]
    to a [`DocList`][docarray.array.doc_list.doc_list.DocList].

    ```python
    from docarray import BaseDoc, DocList, DocVec


    class SimpleDoc(BaseDoc):
        text: str


    dv = DocVec[SimpleDoc]([SimpleDoc(text=f'doc {i}') for i in range(2)])

    dv.to_doc_list().to_csv('simple-dl.csv')
    dv_from_csv = DocList[SimpleDoc].from_csv('simple-dl.csv').to_doc_vec()
    ```

    For more details you can check the [DocList section on CSV serialization](#csv)

### Pandas.Dataframe

- [`from_dataframe()`][docarray.array.doc_vec.io.IOMixinDocVec.from_dataframe] loads a [DocVec][docarray.array.doc_list.doc_list.DocVec] from a [Pandas Dataframe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html).
- [`to_dataframe()`][docarray.array.doc_vec.io.IOMixinDocVec.to_dataframe] saves a [DocVec][docarray.array.doc_list.doc_list.DocVec] to a [Pandas Dataframe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html).

```python
from docarray import BaseDoc, DocVec
from docarray.typing import TorchTensor
import torch


class SimpleDoc(BaseDoc):
    text: str
    tensor: TorchTensor


dv = DocVec[SimpleDoc](
    [SimpleDoc(text=f'doc {i}', tensor=torch.rand(64)) for i in range(2)]
)

df = dv.to_dataframe()
dv_from_dataframe = DocVec[SimpleDoc].from_dataframe(df, tensor_type=TorchTensor)
print(dv_from_dataframe)
```



