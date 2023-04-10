# Serialization for `DocVec`
When sending or storing `DocVec`, you need to use serialization. `DocVec` only supports protobuf to serialize the data.
You can use `to_protobuf()` and `from_protobuf()` to serialize and deserialize a `DocVec`

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

`to_protobuf()` returns a protobuf object of `docarray_pb2.DocVecProto` class. `from_protobuf()` accepts a protobuf message object to construct a `DocVec`.
