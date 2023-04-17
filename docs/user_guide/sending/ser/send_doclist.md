# DocList

When sending or storing [`DocList`][docarray.array.doc_list.doc_list.DocList], you need to use serialization. [`DocList`][docarray.array.doc_list.doc_list.DocList] supports multiple ways to serialize the data.

## JSON

-  [`to_json()`][docarray.array.doc_list.io.IOMixinArray.to_json] serializes a [`DocList`][docarray.array.doc_list.doc_list.DocList] to JSON. It returns the binary representation of the JSON object. 
-  [`from_json()`][docarray.array.doc_list.io.IOMixinArray.from_json] deserializes a [`DocList`][docarray.array.doc_list.doc_list.DocList] from JSON. It can load from either a `str` or `binary` representation of the JSON object.

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

## protobuf

- [`to_protobuf()`][docarray.array.doc_list.io.IOMixinArray.to_protobuf] serializes a [`DocList`][docarray.array.doc_list.doc_list.DocList] to `protobuf`. It returns a `protobuf` object of `docarray_pb2.DocListProto` class.
- [`from_protobuf()`][docarray.array.doc_list.io.IOMixinArray.from_protobuf] deserializes a [`DocList`][docarray.array.doc_list.doc_list.DocList] from `protobuf`. It accepts a `protobuf` message object to construct a [`DocList`][docarray.array.doc_list.doc_list.DocList].

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

## Base64

When transferring data over the network, use `Base64` format to serialize the [`DocList`][docarray.array.doc_list.doc_list.DocList].
Serializing a [`DocList`][docarray.array.doc_list.doc_list.DocList] in Base64 supports both the `pickle` and `protobuf` protocols. You can also choose different compression methods.

- [`to_base64()`][docarray.array.doc_list.io.IOMixinArray.to_base64] serializes a [`DocList`][docarray.array.doc_list.doc_list.DocList] to Base64
- [`from_base64()`][docarray.array.doc_list.io.IOMixinArray.from_base64] deserializes a [`DocList`][docarray.array.doc_list.doc_list.DocList] from Base64:

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

## Binary

- [`save_binary()`][docarray.array.doc_list.io.IOMixinArray.save_binary] saves a [`DocList`][docarray.array.doc_list.doc_list.DocList] to a binary file.
- [`load_binary()`][docarray.array.doc_list.io.IOMixinArray.load_binary] loads a [`DocList`][docarray.array.doc_list.doc_list.DocList] from a binary file.

You can multiple compression methods: `lz4`, `bz2`, `lzma`, `zlib`, and `gzip`.

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

- [to_bytes()][docarray.array.doc_list.io.IOMixinArray.to_bytes] saves a [`DocList`][docarray.array.doc_list.doc_list.DocList] to a byte object.
- [from_bytes()][docarray.array.doc_list.io.IOMixinArray.from_bytes] loads a [`DocList`][docarray.array.doc_list.doc_list.DocList] from a byte object.  

!!! note
    These methods are used under the hood by [save_binary()][docarray.array.doc_list.io.IOMixinArray.to_base64] and [`load_binary()`][docarray.array.doc_list.io.IOMixinArray.load_binary] to prepare/load/save to a binary file. You can also use them directly to work with byte files.

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

## CSV

- [`to_csv()`][docarray.array.doc_list.io.IOMixinArray.to_csv] serializes a [`DocList`][docarray.array.doc_list.doc_list.DocList] to a CSV file.
- [`from_csv()`][docarray.array.doc_list.io.IOMixinArray.from_csv] deserializes a [`DocList`][docarray.array.doc_list.doc_list.DocList] from a CSV file.

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

## Pandas.Dataframe

- [`from_dataframe()`][docarray.array.doc_list.io.IOMixinArray.from_dataframe] loads a [`DocList`][docarray.array.doc_list.doc_list.DocList] from a [Pandas Dataframe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html).
- [`to_dataframe()`][docarray.array.doc_list.io.IOMixinArray.to_dataframe] saves a [`DocList`][docarray.array.doc_list.doc_list.DocList] to a [Pandas Dataframe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html).

```python
from docarray import BaseDoc, DocList


class SimpleDoc(BaseDoc):
    text: str


dl = DocList[SimpleDoc]([SimpleDoc(text=f'doc {i}') for i in range(2)])

df = dl.to_dataframe()
dl_from_dataframe = DocList[SimpleDoc].from_dataframe(df)
print(dl_from_dataframe)
```

See also:

* The serializing [`BaseDoc`](./send_doc.md) section
* The serializing [`DocVec`](./send_docvec.md) section
