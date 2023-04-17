# BaseDoc

You need to serialize a [BaseDoc][docarray.base_doc.doc.BaseDoc] before you can store or send it.

!!! note
    [BaseDoc][docarray.base_doc.doc.BaseDoc] supports serialization to `protobuf` and `json` formats.

## JSON

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

## protobuf

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

See also:

* The serializing [`DocList`](./send_doclist.md) section
* The serializing [`DocVec`](./send_docvec.md) section

