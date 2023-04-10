# Serialization for `DocList`
When sending or storing `DocList`, you need to use serialization. `DocList` supports multiple ways to serialize the data.

## json
You can use `to_json()` and `from_json()` to serialize and deserialize a `DocList`.

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

`to_json()` return the binary representation of the json object. `from_json()` can load from either `str` or `binary` representation of the json object.

```output
b'[{"id":"5540e72d407ae81abb2390e9249ed066","text":"doc 0"},{"id":"fbe9f80d2fa03571e899a2887af1ac1b","text":"doc 1"}]'
```