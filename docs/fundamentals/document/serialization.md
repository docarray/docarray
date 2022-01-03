# Serialization

You can serialize a `Document` into JSON string via {meth}`~jina.types.mixin.ProtoTypeMixin.to_json` or Python dict via {meth}`~jina.types.mixin.ProtoTypeMixin.to_dict` or binary string via {meth}`bytes`:
````{tab} JSON
```python
from jina import Document

Document(content='hello, world', embedding=[1, 2, 3]).to_json()
```

```json
{
  "embedding": [
    1,
    2,
    3
  ],
  "id": "9e36927e576b11ec81971e008a366d48",
  "mime_type": "text/plain",
  "text": "hello, world"
}

```
````

````{tab} Binary
```python
from jina import Document

bytes(Document(content='hello, world', embedding=[1, 2, 3]))
```

```
b'\n aad94436576b11ec81551e008a366d48R\ntext/plainj\x0chello, world\x9a\x01+\n"\n\x18\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x12\x01\x03\x1a\x03<i8\x1a\x05numpy'
```
````

````{tab} Dict
```python
from jina import Document

Document(content='hello, world', embedding=[1, 2, 3]).to_dict()
```

```
{'id': 'c742f7f2576b11ec89aa1e008a366d48', 'mime_type': 'text/plain', 'text': 'hello, world', 'embedding': [1, 2, 3]}
```
````
