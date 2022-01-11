(serialize)=
# Serialization

DocArray is designed to be "ready-to-wire": it assumes you always want to send/receive Document over network across microservices. Hence, serialization of Document is important. This chapter introduces multiple serialization methods of a single Document. 

```{tip}
One should use {ref}`DocumentArray for serializing multiple Documents<docarray-serialization>`, instead of looping over Documents one by one. The former is much faster and yield more compact serialization. 
```


## From/to JSON

```{important}
This feature requires `protobuf` dependency. You can do `pip install "docarray[full]"` to install it.
```

You can serialize a Document as a JSON string via {meth}`~docarray.document.mixins.porting.PortingMixin.to_json`, and then read from it via {meth}`~docarray.document.mixins.porting.PortingMixin.from_json`.

```python
from docarray import Document
import numpy as np

d_as_json = Document(text='hello, world', embedding=np.array([1, 2, 3])).to_json()

d = Document.from_json(d_as_json)

print(d_as_json, d)
```

```text
{
  "embedding": {
    "cls_name": "numpy",
    "dense": {
      "buffer": "AQAAAAAAAAACAAAAAAAAAAMAAAAAAAAA",
      "dtype": "<i8",
      "shape": [
        3
      ]
    }
  },
  "id": "27d4fa4c6d5711ec8c831e008a366d49",
  "mime_type": "text/plain",
  "text": "hello, world"
} 

<Document ('id', 'mime_type', 'text', 'embedding') at 27d4fa4c6d5711ec8c831e008a366d49>
```

(doc-in-bytes)=
## From/to bytes

```{important}
Depending on your values of `protocol` and `compress` arguments, this feature may require `protobuf` and `lz4` dependencies. You can do `pip install "docarray[full]"` to install it.
```


Bytes or binary or buffer, how ever you want to call it, it probably the most common & compact wire format. DocArray provides {meth}`~docarray.document.mixins.porting.PortingMixin.to_bytes` and {meth}`~docarray.document.mixins.porting.PortingMixin.from_bytes` to serialize Document object into bytes.

```python
from docarray import Document
import numpy as np

d = Document(text='hello, world', embedding=np.array([1, 2, 3]))
d_bytes = d.to_bytes()

d_r = Document.from_bytes(d_bytes)

print(d_bytes, d_r)
```

```text
b'\x80\x03cdocarray.document\nDocument\nq\x00)\x81q\x01}q\x02X\x05\x00\x00\x00_dataq\x03cdocarray.document.data\nDocumentData\nq\x04)\x81q\x05}q\x06(X\x0e\x00\x00\x00_reference_docq\x07h\x01X\x02\x00\x00\x00idq\x08X \x00\x00\x005d29a9f26d5911ec88d51e008a366d49q\tX\t\x00\x00\x00parent_...

<Document ('id', 'mime_type', 'text', 'embedding') at 3644c0fa6d5a11ecbb081e008a366d49>
```

Default serialization protocol is `pickle`, you can change it to `protobuf` by specifying `.to_bytes(protocol='protobuf')`. You can also add compression to it and make the result bytes smaller. For example, 

```python
d = Document(text='hello, world', embedding=np.array([1, 2, 3]))
print(len(d.to_bytes(protocol='protobuf', compress='gzip')))
```

gives:

```text
110
```

whereas the default `.to_bytes()` gives `666` (spooky~).

Note that when deserializing from a non-default binary serialization, you need to specify the correct `protocol` and `compress` arguments used at the serialization time:

```python
d = Document.from_bytes(d_bytes, protocol='protobuf', compress='gzip')
```

```{tip}
If you go with default `protcol` and `compress` settings, you can simply use `bytes(d)`, which is more Pythonic.
```


## From/to base64

```{important}
Depending on your values of `protocol` and `compress` arguments, this feature may require `protobuf` and `lz4` dependencies. You can do `pip install "docarray[full]"` to install it.
```

In some cases such as in REST API, you are allowed only to send/receive string not bytes. You can serialize Document into base64 string via {meth}`~docarray.document.mixins.porting.PortingMixin.to_base64` and load it via {meth}`~docarray.document.mixins.porting.PortingMixin.from_base64`.

```python
from docarray import Document
d = Document(text='hello', embedding=[1, 2, 3])

print(d.to_base64())
```

```text
gANjZG9jYXJyYXkuZG9jdW1lbnQKRG9jdW1lbnQKcQApgXEBfXECWAUAAABfZGF0YXEDY2RvY2FycmF5LmRvY3VtZW50LmRhdGEKRG9jdW1lbnREYXRhCnEEKYFxBX1xBihYDgAAAF9yZWZlcmVuY2VfZG9jcQdoAVgCAAAAaWRxCFggAAAAZmZjNTY3ODg3MzAyMTFlY2E4NjMxZTAwOGEzNjZkNDlxCVgJAAAAcGFyZW50X2lkcQpOWAsAAABncmFudWxhcml0eXELTlgJAAAAYWRqYWNlbmN5cQxOWAYAAABidWZmZXJxDU5YBAAAAGJsb2JxDk5YCQAAAG1pbWVfdHlwZXEPWAoAAAB0ZXh0L3BsYWlucRBYBAAAAHRleHRxEVgFAAAAaGVsbG9xElgHAAAAY29udGVudHETTlgGAAAAd2VpZ2h0cRROWAMAAAB1cmlxFU5YBAAAAHRhZ3NxFk5YBgAAAG9mZnNldHEXTlgIAAAAbG9jYXRpb25xGE5YCQAAAGVtYmVkZGluZ3EZXXEaKEsBSwJLA2VYCAAAAG1vZGFsaXR5cRtOWAsAAABldmFsdWF0aW9uc3EcTlgGAAAAc2NvcmVzcR1OWAYAAABjaHVua3NxHk5YBwAAAG1hdGNoZXNxH051YnNiLg==
```

You can set `protocol` and `compress` to get a more compact string:

```python
from docarray import Document
d = Document(text='hello', embedding=[1, 2, 3])

print(len(d.to_base64()))
print(len(d.to_base64(protocol='protobuf', compress='lz4')))
```

```text
664
156
```

Note that the same `protocol` and `compress` must be followed when using `.from_base64`.


## From/to dict

```{important}
This feature requires `protobuf` dependency. You can do `pip install "docarray[full]"` to install it.
```

You can serialize a Document as a Python `dict` via {meth}`~docarray.document.mixins.porting.PortingMixin.to_dict`, and then read from it via {meth}`~docarray.document.mixins.porting.PortingMixin.from_dict`.

```python
from docarray import Document
import numpy as np

d_as_dict = Document(text='hello, world', embedding=np.array([1, 2, 3])).to_dict()

d = Document.from_dict(d_as_dict)

print(d_as_dict, d)
```

```text
{'id': 'b29d39066d5611ec87661e008a366d49', 'text': 'hello, world', 'mime_type': 'text/plain', 'embedding': {'dense': {'buffer': 'AQAAAAAAAAACAAAAAAAAAAMAAAAAAAAA', 'shape': [3], 'dtype': '<i8'}, 'cls_name': 'numpy'}} 

<Document ('id', 'mime_type', 'text', 'embedding') at b29d39066d5611ec87661e008a366d49>
```

(strict-arg-explain)=
```{note}
Note that the result dict is very "stricted" in the sense that all fields and values boil down to very basic data type such as `int`, `float`, `string`. This behavior is designed due to the "serialization to `dict`" is often an intermediate step of serializing into JSON/YAML. Hence all values in `dict` must be schema-friendly. After all, a Python `dict` object means nothing if you are not working in Python. 

You can use `to_dict(strict=False)` to override this behavior. This will preserve the original Python data type of every value, which may not be JSON-friendly. But hey, you want it.   
```

## From/to Protobuf

```{important}
This feature requires `protobuf` dependency. You can do `pip install "docarray[full]"` to install it.
```

You can also serialize a Document object into a Protobuf Message object. This is less frequently used as it is often an intermediate step when serializing into bytes, as in `to_dict()`. However, if you work with Python Protobuf API, having a Python Protobuf Message object at hand can be useful.


```python
from docarray import Document

d_proto = Document(uri='apple.jpg').to_protobuf()
print(type(d_proto), d_proto)
d = Document.from_protobuf(d_proto)
```

```text
<class 'docarray_pb2.DocumentProto'> 

id: "d66463b46d6a11ecbf891e008a366d49"
uri: "apple.jpg"
mime_type: "image/jpeg"

<Document ('id', 'mime_type', 'uri') at e4b215106d6a11ecb28b1e008a366d49>
```

One can refer to the [Protobuf specification of `Document`](../../proto/index.md) for details.  


## What's next?

Serializing single Document can be useful but often we want to do things in bulk, say hundreds or one million Documents at once. In that case, looping over each Document and serializing one by one is inefficient. In DocumentArray, we will introduce the similar interfaces {meth}`~docarray.array.mixins.io.binary.BinaryIOMixin.to_bytes`, {meth}`~docarray.array.mixins.io.json.JsonIOMixin.to_json`, and {meth}`~docarray.array.mixins.io.json.JsonIOMixin.to_list` that allows one to [serialize multiple Documents much faster and more compact](../documentarray/serialization.md).