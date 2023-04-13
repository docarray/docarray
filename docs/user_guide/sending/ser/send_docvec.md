# DocVec

When sending or storing [`DocVec`][docarray.array.doc_list.doc_list.DocVec], you need to use serialization. [DocVec][docarray.array.doc_list.doc_list.DocVec] only supports protobuf to serialize the data.
You can use [`to_protobuf`][docarray.array.doc_list.doc_list.DocVec.to_protobuf] and [`from_protobuf`][docarray.array.doc_list.doc_list.DocVec.from_protobuf] to serialize and deserialize a [DocVec][docarray.array.doc_list.doc_list.DocVec]

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

!!! note
    We are planning to add more serialization formats in the future, notably JSON.

[`to_protobuf`][docarray.array.doc_list.doc_list.DocVec.to_protobuf] returns a protobuf object of `docarray_pb2.DocVecProto` class. [`from_protobuf`][docarray.array.doc_list.doc_list.DocVec.from_protobuf] accepts a protobuf message object to construct a [DocVec][docarray.array.doc_list.doc_list.DocVec].

* The serializing [BaseDoc](./send_doc.md) section
* The serializing [DocList](./send_doclist.md) section
