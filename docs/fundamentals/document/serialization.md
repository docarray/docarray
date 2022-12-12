(serialize)=
# Serialization

DocArray is designed to be "ready-to-wire": it assumes you always want to send/receive Documents over the network across microservices. This chapter introduces a Document's multiple serialization methods. 

```{tip}
You should use {ref}`DocumentArray for serializing multiple Documents<docarray-serialization>`, instead of looping over Documents one by one. The former is much faster and yields more compact serialization. 
```

```{hint}
You may wonder: why isn't serialization part of constructor? Both are similar. Nonetheless, serialization often contains elements that don't really fit into constructor: input & output model, data schema, compression, extra-dependencies. We made a decision with DocArray to separate the constructor and serialization for the sake of clarity and maintainability. 
```

(doc-json)=
## From/to JSON

```{tip}
If you're building a webservice and want to use JSON for passing DocArray objects, then data validation and field-filtering can be crucial. In this case, we highly recommend checking out {ref}`fastapi-support` and following the methods there.   
```

```{important}
Depending on which protocol you use, this feature requires `pydantic` or `protobuf` dependency. You can do `pip install "docarray[full]"` to install it.
```



You can serialize a Document as a JSON string with {meth}`~docarray.document.mixins.porting.PortingMixin.to_json`, and then read from it with {meth}`~docarray.document.mixins.porting.PortingMixin.from_json`.

```python
from docarray import Document
import numpy as np

d_as_json = Document(text='hello, world', embedding=np.array([1, 2, 3])).to_json()

d = Document.from_json(d_as_json)

print(d_as_json, d)
```

```text
{"id": "641032d677b311ecb67a1e008a366d49", "parent_id": null, "granularity": null, "adjacency": null, "blob": null, "tensor": null, "mime_type": "text/plain", "text": "hello, world", "weight": null, "uri": null, "tags": null, "offset": null, "location": null, "embedding": [1, 2, 3], "modality": null, "evaluations": null, "scores": null, "chunks": null, "matches": null} 

<Document ('id', 'mime_type', 'text', 'embedding') at 641032d677b311ecb67a1e008a366d49>
```

By default, Documents use {ref}`JSON Schema and pydantic model<schema-gen>` for serialization, i.e. `protocol='jsonschema'`. To use Protobuf as the JSON serialization backend, pass `protocol='protobuf'` to the method:


```python
from docarray import Document

d = Document(text='hello, world', embedding=np.array([1, 2, 3]))
d.to_json(protocol='protobuf')
```

```json
{
  "id": "db66bc2e77b311eca5f51e008a366d49",
  "text": "hello, world",
  "mimeType": "text/plain",
  "embedding": {
    "dense": {
      "buffer": "AQAAAAAAAAACAAAAAAAAAAMAAAAAAAAA",
      "shape": [
        3
      ],
      "dtype": "<i8"
    },
    "clsName": "numpy"
  }
}
```

When using a RESTful API, you should use `protocol='jsonschema'` as the resulting JSON will follow a pre-defined schema. This is highly appreciated for modern webservice engineering.

Note that you can pass extra arguments to control field inclusion/exclusion, lower/uppercase. For example, you can remove fields that are empty or `none` with:

```python
from docarray import Document

d = Document(text='hello, world', embedding=[1, 2, 3])
d.to_json(exclude_none=True)
```

```json
{"id": "cdbc4f7a77b411ec96ad1e008a366d49", "mime_type": "text/plain", "text": "hello, world", "embedding": [1, 2, 3]}
```

This is easier on the eyes. But when building a RESTful API, you don't need to explicitly do this -- the pydantic model handles everything for you. More information can be found in {ref}`fastapi-support`.

```{seealso}
To find out what extra parameters you can pass to `to_json()`/`to_dict()`, check out:
- [`protocol='jsonschema', **kwargs`](https://pydantic-docs.helpmanual.io/usage/exporting_models/#modeljson) 
- [`protocol='protobuf', **kwargs`](https://googleapis.dev/python/protobuf/latest/google/protobuf/json_format.html#google.protobuf.json_format.MessageToJson)
```


(arbitrary-json)=

### From/to arbitrary JSON

Arbitrary JSON is unschema-ed JSON. It often comes from handcrafted JSON, or an export file from other libraries. Its schema is unknown to DocArray, so by principle you can't load it.

But principles be damned. To load an arbitrary JSON file set `protocol=None`. 

As an _arbitrary_ JSON, don't expect it to work smoothly. DocArray will try its best to parse the fields: by first loading the JSON into a `dict` object; and then building a Document with `Document(dict)`; when encountering unknown attributes it follows the behavior {ref}`described here<unk-attribute>`.

As a rule of thumb, if you only work inside DocArray's ecosystem, always use schema-ed JSON (`.to_json(protocol='jsonschema')`, or `.to_json(protocol='protobuf')`) over unschema-ed JSON. If you're exporting DocArray's JSON to other ecosystems, also use schema-ed JSON. Your engineer friends will appreciate it as it is easier to integrate. In fact, DocArray does **not** offer unschema-ed JSON export, so your engineer friends will never be upset.

Read more about {ref}`schema-gen` support in DocArray.


(doc-in-bytes)=
## From/to bytes

```{important}
Depending on your `protocol` and `compress` argument values, this feature may require `protobuf` and `lz4` dependencies. You can run `pip install "docarray[full]"` to install it.
```

Bytes or binary or buffer, however you want to call it, is probably the most common and compact wire format. DocArray provides {meth}`~docarray.document.mixins.porting.PortingMixin.to_bytes` and {meth}`~docarray.document.mixins.porting.PortingMixin.from_bytes` to serialize Document objects into bytes.

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

The default serialization protocol is `pickle` -- you can change it to `protobuf` by specifying `.to_bytes(protocol='protobuf')`. You can also add compression to make the resulting bytes smaller:

```python
d = Document(text='hello, world', embedding=np.array([1, 2, 3]))
print(len(d.to_bytes(protocol='protobuf', compress='gzip')))
```

gives:

```text
110
```

whereas the default `.to_bytes()` gives `666`.

Note that when deserializing from a non-default binary serialization, you need to specify the correct `protocol` and `compress` arguments used at serialization time:

```python
d = Document.from_bytes(d_bytes, protocol='protobuf', compress='gzip')
```

```{tip}
If you go with default `protcol` and `compress` settings, you can simply use `bytes(d)`, which is more Pythonic.
```


## From/to base64

```{important}
Depending on your `protocol` and `compress` argument values, this feature may require `protobuf` and `lz4` dependencies. You can run `pip install "docarray[full]"` to install it.
```

Sometimes, such as with RESTful APIs, you can only send/receive strings, not bytes. You can serialize a Document into a base64 string with {meth}`~docarray.document.mixins.porting.PortingMixin.to_base64` and load it with {meth}`~docarray.document.mixins.porting.PortingMixin.from_base64`.

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

Note that you must follow the same `protocol` and `compress` when using `.from_base64`.

(doc-dict)=
## From/to dict

```{important}
This feature requires the `protobuf` or `pydantic` dependency. You can run `pip install "docarray[full]"` to install it.
```

You can serialize a Document as a Python `dict` with {meth}`~docarray.document.mixins.porting.PortingMixin.to_dict`, and then read from it with {meth}`~docarray.document.mixins.porting.PortingMixin.from_dict`.

```python
from docarray import Document
import numpy as np

d_as_dict = Document(text='hello, world', embedding=np.array([1, 2, 3])).to_dict()

d = Document.from_dict(d_as_dict)

print(d_as_dict, d)
```

```text
{'id': '5596c84c77b711ecafed1e008a366d49', 'parent_id': None, 'granularity': None, 'adjacency': None, 'blob': None, 'tensor': None, 'mime_type': 'text/plain', 'text': 'hello, world', 'weight': None, 'uri': None, 'tags': None, 'offset': None, 'location': None, 'embedding': [1, 2, 3], 'modality': None, 'evaluations': None, 'scores': None, 'chunks': None, 'matches': None} 

<Document ('id', 'mime_type', 'text', 'embedding') at 5596c84c77b711ecafed1e008a366d49>
```

As the intermediate step of `to_json()`/`from_json()` it's unlikely you'll use dict IO directly. Nonetheless, you can pass the same `protocol` and `kwargs` as described in {ref}`doc-json` to control serialization behavior.

## From/to Protobuf

```{important}
This feature requires `protobuf` dependency. You can run `pip install "docarray[full]"` to install it.
```

You can also serialize a Document object into a Protobuf Message object. This is used less frequently as it's often an intermediate step when serializing into bytes, as in `to_dict()`. However, if you work with Python's Protobuf API, having a Python Protobuf Message object at hand can be useful.


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

Refer to the [Protobuf specification of `Document`](../../proto/index.md) for details.  

When `.tensor` or `.embedding` contains a framework-specific ndarray-like object, you can use `.to_protobuf(..., ndarray_type='numpy')` or `.to_protobuf(..., ndarray_type='list')` to cast them into `list` or `numpy.ndarray` automatically. This helps ensure maximum compatibility between different microservices.

## What's next?

Serializing a single Document can be useful, but often you want to do things in bulk, say one hundred or one million Documents at once. In that case, looping over each Document and serializing one by one is inefficient. In DocumentArray, we introduce the similar interfaces {meth}`~docarray.array.mixins.io.binary.BinaryIOMixin.to_bytes`, {meth}`~docarray.array.mixins.io.json.JsonIOMixin.to_json`, and {meth}`~docarray.array.mixins.io.json.JsonIOMixin.to_list` that let you [serialize multiple Documents more quickly and compactly](../documentarray/serialization.md).
