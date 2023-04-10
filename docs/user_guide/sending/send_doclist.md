# Serialization for `DocList`
When sending or storing `DocList`, you need to use serialization. `DocList` supports multiple ways to serialize the data.

## JSON
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

`to_json()` returns the binary representation of the json object. `from_json()` can load from either `str` or `binary` representation of the json object.

```output
b'[{"id":"5540e72d407ae81abb2390e9249ed066","text":"doc 0"},{"id":"fbe9f80d2fa03571e899a2887af1ac1b","text":"doc 1"}]'
```

## Protobuf
When using protobuf, you can use `to_protobuf()` and `from_protobuf()` to serialize and deserialize a `DocList`

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

`to_protobuf()` returns a protobuf object of `docarray_pb2.DocListProto` class. `from_protobuf()` accepts a protobuf message object to construct a `DocList`.

## Base64
When transferring over the network, you can choose `Base64` format to serialize the `DocList`.
Storing a `DocList` in Base64 supports both `pickle` and `protobuf` protocols. Besides, you can choose different compression methods.


| Compression Methods | Notes |
| --- |-------|
| `lz4` |       |
| `bz2` |       |
| `lzma` |       |
| `zlib` |       |
| `gzip` |       |

```python
from docarray import BaseDoc, DocList


class SimpleDoc(BaseDoc):
    text: str


dl = DocList[SimpleDoc]([SimpleDoc(text=f'doc {i}') for i in range(2)])

base64_repr_dl = dl.to_base64(compress=None, protocol='pickle')

dl_from_base64 = DocList[SimpleDoc].to_base64(base64_repr_dl , compress=None, protocol='pickle')
```

## Binary
Similar as in `Base64` serialization, `Binary` serialization also supports different protocols and compression methods.

```python
from docarray import BaseDoc, DocList


class SimpleDoc(BaseDoc):
    text: str


dl = DocList[SimpleDoc]([SimpleDoc(text=f'doc {i}') for i in range(2)])

dl.save_binary('simple-dl.pickle', compress=None, protocol='pickle')

dl_from_binary = DocList[SimpleDoc].load_binary('simple-dl.pickle', compress=None, protocol='pickle')
```

The `DocList` is stored at `simple-dl.pickle` file.

### Bytes
Under the hood, `save_binary()` prepares the file object and calls `to_bytes()` function to convert the `DocList` into a byte object. You can use `to_bytes()` function directly and use `from_bytes()` to load the `DocList` from a byte object. You can use `protocol` to choose between `pickle` and `protobuf`. Besides, `to_bytes()`  and `save_bytes()` support multiple options for `compress` as well. 

```python
from docarray import BaseDoc, DocList


class SimpleDoc(BaseDoc):
    text: str


dl = DocList[SimpleDoc]([SimpleDoc(text=f'doc {i}') for i in range(2)])

bytes_dl = dl.to_bytes(protocol='pickle', compress=None)

dl_from_bytes = DocList[SimpleDoc].from_bytes(bytes_dl, compress=None, protocol='pickle')
```


## CSV
You can use `from_csv()` and `to_csv()` to de-/serializae and deserialize the `DocList` from/to a CSV file. Use the `dialect` parameter to choose the dialect of the CSV format. Check more details in the API doc. TODO: Add api doc here.

```python
from docarray import BaseDoc, DocList


class SimpleDoc(BaseDoc):
    text: str


dl = DocList[SimpleDoc]([SimpleDoc(text=f'doc {i}') for i in range(2)])

dl.to_csv('simple-dl.csv')
dl_from_csv = DocList[SimpleDoc].from_csv('simple-dl.csv')
print(dl_from_csv)
```


## Pandas.Dataframe
You can use `from_pandas()` and `to_pandas()` to load/save the `DocList` from/to a pandas DataFrame.

```python
from docarray import BaseDoc, DocList


class SimpleDoc(BaseDoc):
    text: str


dl = DocList[SimpleDoc]([SimpleDoc(text=f'doc {i}') for i in range(2)])

df = dl.to_pandas()
dl_from_dataframe = DocList[SimpleDoc].from_pandas(df)
print(dl_from_dataframe)
```