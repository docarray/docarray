# DocVec

When sending or storing [`DocVec`][docarray.array.doc_list.doc_list.DocVec], you need to use protobuf serialization. 

!!! note
    We plan to add more serialization formats in the future, notably JSON.

## protobuf

- [`to_protobuf`][docarray.array.doc_list.doc_list.DocVec.to_protobuf] serializes a [DocVec][docarray.array.doc_list.doc_list.DocVec] to `protobuf`. It returns a `protobuf` object of `docarray_pb2.DocVecProto` class. 
- [`from_protobuf`][docarray.array.doc_list.doc_list.DocVec.from_protobuf] deserializes a [DocVec][docarray.array.doc_list.doc_list.DocVec] from `protobuf`. It accepts a protobuf message object to construct a [DocVec][docarray.array.doc_list.doc_list.DocVec].

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

## See also

* The serializing [`BaseDoc`](./send_doc.md) section
* The serializing [`DocList`](./send_doclist.md) section
