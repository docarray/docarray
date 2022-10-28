(docarray-serialization)=
# Serialization

DocArray is designed to be "ready-to-wire" at anytime. Serialization is important.
DocumentArray provides multiple serialization methods that allows one transfer DocumentArray object over network and across different microservices.
Moreover, there is the ability to store/load `DocumentArray` objects to/from disk.

- JSON string: `.from_json()`/`.to_json()`
  - Pydantic model: `.from_pydantic_model()`/`.to_pydantic_model()`
- Bytes (compressed): `.from_bytes()`/`.to_bytes()`
  - Disk serialization: `.save_binary()`/`.load_binary()`
- Base64 (compressed): `.from_base64()`/`.to_base64()` 
- Protobuf Message: `.from_protobuf()`/`.to_protobuf()`
- Python List: `.from_list()`/`.to_list()`
- Pandas Dataframe: `.from_dataframe()`/`.to_dataframe()`
- Cloud: `.push()`/`.pull()`




## From/to JSON


```{tip}
If you are building a webservice and want to use JSON for passing DocArray objects, then data validation and field-filtering can be crucial. In this case, it is highly recommended to check out {ref}`fastapi-support` and follow the methods there.   
```

```{important}
Depending on which protocol you use, this feature requires `pydantic` or `protobuf` dependency. You can do `pip install "docarray[common]"` to install it.
```



```python
from docarray import DocumentArray, Document

da = DocumentArray([Document(text='hello'), Document(text='world')])
da.to_json()
```

```text
[{"id": "a677577877b611eca3811e008a366d49", "parent_id": null, "granularity": null, "adjacency": null, "blob": null, "tensor": null, "mime_type": "text/plain", "text": "hello", "weight": null, "uri": null, "tags": null, "offset": null, "location": null, "embedding": null, "modality": null, "evaluations": null, "scores": null, "chunks": null, "matches": null}, {"id": "a67758f477b611eca3811e008a366d49", "parent_id": null, "granularity": null, "adjacency": null, "blob": null, "tensor": null, "mime_type": "text/plain", "text": "world", "weight": null, "uri": null, "tags": null, "offset": null, "location": null, "embedding": null, "modality": null, "evaluations": null, "scores": null, "chunks": null, "matches": null}]
```


```python
da_r = DocumentArray.from_json(da.to_json())

da_r.summary()
```

```text
                  Documents Summary                   
                                                      
  Length                 2                            
  Homogenous Documents   True                         
  Common Attributes      ('id', 'mime_type', 'text')  
                                                      
                     Attributes Summary                     
                                                            
  Attribute   Data type   #Unique values   Has empty value  
 ────────────────────────────────────────────────────────── 
  id          ('str',)    2                False            
  mime_type   ('str',)    1                False            
  text        ('str',)    2                False            

```


```{seealso}
To load an arbitrary JSON file, please set `protocol=None` {ref}`as descrbied here<arbitrary-json>`.

More parameters and usages can be found in the Document-level {ref}`doc-json`.
```


## From/to bytes

```{important}
Depending on your values of `protocol` and `compress` arguments, this feature may require `protobuf` and `lz4` dependencies. You can do `pip install "docarray[full]"` to install it.
```

Serialization into bytes often yield more compact representation than in JSON. Similar to {ref}`the Document serialization<doc-in-bytes>`, DocumentArray can be serialized with different `protocol` and `compress` combinations. In its most simple form,

```python
from docarray import DocumentArray, Document

da = DocumentArray([Document(text='hello'), Document(text='world')])
da.to_bytes()
```

```text
b'\x80\x03cdocarray.array.document\nDocumentArray\nq\x00)\x81q\x01}q\x02(X\x05\x00\x00\x00_dataq\x03]q\x04(cdocarray.document\nDocument\nq\x05) ...
```

```python
da_r = DocumentArray.from_bytes(da.to_bytes())

da_r.summary()
```

```text
                  Documents Summary                   
                                                      
  Length                 2                            
  Homogenous Documents   True                         
  Common Attributes      ('id', 'mime_type', 'text')  
                                                      
                     Attributes Summary                     
                                                            
  Attribute   Data type   #Unique values   Has empty value  
 ────────────────────────────────────────────────────────── 
  id          ('str',)    2                False            
  mime_type   ('str',)    1                False            
  text        ('str',)    2                False      
```

```{tip}
If you go with default `protcol` and `compress` settings, you can simply use `bytes(da)`, which is more Pythonic.
```

The table below summarize the supported serialization protocols and compressions:

| `protocol=...`           | Description                                                                                          | Remarks                                                                                                                     |
|--------------------------|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| `pickle-array` (default) | Serialize the whole array in one-shot using Python `pickle`                                          | Often fastest. Not portable to other languages. Insecure in production.                                                     |
| `protobuf-array`         | Serialize the whole array using [`DocumentArrayProto`](../../../proto/#docarray.DocumentArrayProto). | Portable to other languages if they implement `DocumentArrayProto`. 2GB max-size (pre-compression) restriction by Protobuf. |
| `pickle`                 | Serialize elements one-by-one using Python `pickle`.                                                 | Allow streaming. Not portable to other languages. Insecure in production.                                                   |
| `protobuf`               | Serialize elements one-by-one using [`DocumentProto`](../../../proto/#docarray.DocumentProto).       | Allow streaming. Portable to other languages if they implement `DocumentProto`. No max-size restriction                     |

For compressions, the following algorithms are supported: `lz4`, `bz2`, `lzma`, `zlib`, `gzip`. The most frequently used ones are `lz4` (fastest) and `gzip` (most widely used).

If you specified non-default `protocol` and `compress` in {meth}`~docarray.array.mixins.io.binary.BinaryIOMixin.to_bytes`, you will need to specify the same in {meth}`~docarray.array.mixins.io.binary.BinaryIOMixin.from_bytes`.

Depending on the use cases, you can choose the one works best for you. Here is a benchmark on serializing a DocumentArray with one million near-empty Documents (i.e. init with `DocumentArray.empty(...)` where each Document has only `id`).

```{figure} images/benchmark-size.svg
```

```{figure} images/benchmark-time.svg
```

The benchmark was conducted [on the codebase of Jan. 5, 2022](https://github.com/jina-ai/docarray/tree/a56067e486d2318e05bcf6088bd1436040107ad2).  

Depending on how you want to interpret the results, the figures above can be an over-estimation/under-estimation of the serialization latency: one may argue that near-empty Documents are not realistic, but serializing a DocumentArray with one million Documents is also unreal. In practice, DocumentArray passing across microservices are relatively small, say at thousands, for better overlapping the network latency and computational overhead.

(wire-format)=
### Wire format of `pickle` and `protobuf`

When set `protocol=pickle` or `protobuf`, the resulting bytes look like the following:

```text
--------------------------------------------------------------------------------------------------------
|   version    |   len(docs)    |  doc1_bytes  |  doc1.to_bytes()  |  doc2_bytes  |  doc2.to_bytes() ...
---------------------------------------------------------------------------------------------------------
| Fixed-length |  Fixed-length  | Fixed-length |  Variable-length  | Fixed-length |  Variable-length ...
--------------------------------------------------------------------------------------------------------
      |               |               |                  |                 |               |
    uint8           uint64          uint32        Variable-length         ...             ...

```

Here `version` is a `uint8` that specifies the serialization version of the `DocumentArray` serialization format, followed by `len(docs)` which is a `uint64` that specifies the amount of serialized documents.
Afterwards, `doc1_bytes` describes how many bytes are used to serialize `doc1`, followed by `doc1.to_bytes()` which is the bytes data of the document itself.
The pattern `dock_bytes` and `dock.to_bytes` is repeated `len(docs)` times.


### From/to disk

If you want to store a `DocumentArray` to disk you can use `.save_binary(filename, protocol, compress)` where `protocol` and `compress` refer to the protocol and compression methods used to serialize the data.
If you want to load a `DocumentArray` from disk you can use `.load_binary(filename, protocol, compress)`.

For example, the following snippet shows how to save/load a `DocumentArray` in `my_docarray.bin`.

```python
from docarray import DocumentArray, Document

da = DocumentArray([Document(text='hello'), Document(text='world')])

da.save_binary('my_docarray.bin', protocol='protobuf', compress='lz4')
da_rec = DocumentArray.load_binary(
    'my_docarray.bin', protocol='protobuf', compress='lz4'
)
da_rec.summary()
```

```text
                  Documents Summary                   
                                                      
  Length                 2                            
  Homogenous Documents   True                         
  Common Attributes      ('id', 'mime_type', 'text')  
                                                      
                     Attributes Summary                     
                                                            
  Attribute   Data type   #Unique values   Has empty value  
 ────────────────────────────────────────────────────────── 
  id          ('str',)    2                False            
  mime_type   ('str',)    1                False            
  text        ('str',)    2                False            
                                                                                               
```


User do not need  to remember the protocol and compression methods on loading. You can simply specify `protocol` and `compress` in the file extension via:

```text
filename.protobuf.gzip
         ~~~~~~~~ ^^^^
             |      |
             |      |-- compress
             |
             |-- protocol
```


When a filename is given as the above format in `.save_binary`, you can simply load it back with `.load_binary` without specifying the protocol and compress method again.


The previous code snippet can be simplified to 

```python
da.save_binary('my_docarray.protobuf.lz4')
da_rec = DocumentArray.load_binary('my_docarray.protobuf.lz4')
```


### Stream large binary serialization from disk

In particular, if a serialization uses `protocol='pickle'` or `protocol='protobuf'`, then you can load it via streaming with a constant memory consumption by setting `streaming=True`:

```python
from docarray import DocumentArray, Document

da_generator = DocumentArray.load_binary(
    'xxxl.bin', protocol='pickle', compress='gzip', streaming=True
)

for d in da_generator:
    d: Document
    # go nuts with `d`
```


## From/to base64

```{important}
Depending on your values of `protocol` and `compress` arguments, this feature may require `protobuf` and `lz4` dependencies. You can do `pip install "docarray[full]"` to install it.
```

Serialize into base64 can be useful when binary string is not allowed, e.g. in REST API. This can be easily done via {meth}`~docarray.array.mixins.io.binary.BinaryIOMixin.to_base64` and {meth}`~docarray.array.mixins.io.binary.BinaryIOMixin.from_base64`. Like in binary serialization, one can specify `protocol` and `compress`:

```python
from docarray import DocumentArray

da = DocumentArray.empty(10)

d_str = da.to_base64(protocol='protobuf', compress='lz4')
print(len(d_str), d_str)
```

```text
176 BCJNGEBAwHUAAAD/Iw+uQdpL9UDNsfvomZb8m7sKIGRkNTIyOTQyNzMwMzExZWNiM2I1MWUwMDhhMzY2ZDQ5MgAEP2FiNDIAHD9iMTgyAB0vNWUyAB0fYTIAHh9myAAdP2MzYZYAHD9jODAyAB0fZDIAHT9kMTZkAABQNjZkNDkAAAAA
```

To deserialize, remember to set the correct `protocol` and `compress`:

```python
from docarray import DocumentArray

da = DocumentArray.from_base64(d_str, protocol='protobuf', compress='lz4')
da.summary()
```

```text
  Length                 10       
  Homogenous Documents   True     
  Common Attributes      ('id',)  
                                  
                     Attributes Summary                     
                                                            
  Attribute   Data type   #Unique values   Has empty value  
 ────────────────────────────────────────────────────────── 
  id          ('str',)    10               False                                                                    
```

## From/to Protobuf

Serializing to Protobuf Message is less frequently used, unless you are using Python Protobuf API. Nonetheless, you can use {meth}`~docarray.array.mixins.io.binary.BinaryIOMixin.from_protobuf` and {meth}`~docarray.array.mixins.io.binary.BinaryIOMixin.to_protobuf` to get a Protobuf Message object in Python.

```python
from docarray import DocumentArray, Document

da = DocumentArray([Document(text='hello'), Document(text='world')])
da.to_bytes()
```

```text
docs {
  id: "2571b8b66e4d11ec9f271e008a366d49"
  text: "hello"
  mime_type: "text/plain"
}
docs {
  id: "2571ba466e4d11ec9f271e008a366d49"
  text: "world"
  mime_type: "text/plain"
}
```

## From/to list

```{important}
This feature requires `protobuf` or `pydantic` dependency. You can do `pip install "docarray[full]"` to install it.
```

Serializing to/from Python list is less frequently used for the same reason as `Document.to_dict()`: it is often an intermediate step of serializing to JSON. You can do:

```python
from docarray import DocumentArray, Document

da = DocumentArray([Document(text='hello'), Document(text='world')])
da.to_list()
```

```text
[{'id': 'ae55782a6e4d11ec803c1e008a366d49', 'text': 'hello', 'mime_type': 'text/plain'}, {'id': 'ae557a146e4d11ec803c1e008a366d49', 'text': 'world', 'mime_type': 'text/plain'}]
```

```{seealso}
More parameters and usages can be found in the Document-level {ref}`doc-dict`.
```

## From/to dataframe

```{important}
This feature requires `pandas` dependency. You can do `pip install "docarray[full]"` to install it.
```

One can convert between a DocumentArray object and a `pandas.dataframe` object.

```python
from docarray import DocumentArray, Document

da = DocumentArray([Document(text='hello'), Document(text='world')])
da.to_dataframe()
```

```text
                                 id   text   mime_type
0  43cb93b26e4e11ec8b731e008a366d49  hello  text/plain
1  43cb95746e4e11ec8b731e008a366d49  world  text/plain
```

To build a DocumentArray from dataframe,

```python
df = ...
da = DocumentArray.from_dataframe(df)
```

## From/to cloud

```{important}
This feature requires `rich` and `requests` dependency. You can do `pip install "docarray[full]"` to install it.
```

{meth}`~docarray.array.mixins.io.pushpull.PushPullMixin.push` and {meth}`~docarray.array.mixins.io.pushpull.PushPullMixin.pull` allows you to serialize a DocumentArray object to Jina Cloud and share it across machines.

Considering you are working on a GPU machine via Google Colab/Jupyter. After preprocessing and embedding, you got everything you need in a DocumentArray. You can easily store it to the cloud via:

```python
from docarray import DocumentArray

da = DocumentArray(...)  # heavylifting, processing, GPU task, ...
da.push('myda123', show_progress=True)
```

```{figure} images/da-push.png
```

Then on your local laptop, simply pull it:

```python
from docarray import DocumentArray

da = DocumentArray.pull('myda123', show_progress=True)
```

Now you can continue the work at local, analyzing `da` or visualizing it. Your friends & colleagues who know the token `myda123` can also pull that DocumentArray. It's useful when you want to quickly share the results with your colleagues & friends.

The maximum size of an upload is 4GB under the `protocol='protobuf'` and `compress='gzip'` setting. The lifetime of an upload is one week after its creation.

To avoid unnecessary download when upstream DocumentArray is unchanged, you can add `DocumentArray.pull(..., local_cache=True)`.

