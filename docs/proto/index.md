```{include} docs.md
```

## 

`````{admonition} Note
:class: caution
As `tags` in Document does not have a fixed schema, it is declared with type `google.protobuf.Struct` in the 
`DocumentProto` protobuf declaration. However, `google.protobuf.Struct` follows the JSON specification and does not 
differentiate `int` from `float`. **So, data of type `int` in `tags` will be casted to `float` when request is
sent to executor.**
As a result, users need be explicit and cast the data to the expected type as follows.
````{tab} ✅ Do
```{code-block} python
---
emphasize-lines: 9, 10
---
animals = ['cat', 'dog', 'fish']

da = DocumentArray([Document(id=i) for i in range(3)])

da.save_binary('aux.bin', protocol='protobuf')
da_loaded = DocumentArray.load_binary('aux.bin', protocol='protobuf')

for doc in da_loaded:
    index = int(da.tags['id'])
    print(animals[index])
```
````
````{tab} 😔 Don't
```{code-block} python
---
emphasize-lines: 9, 10
---
animals = ['cat', 'dog', 'fish']

da = DocumentArray([Document(id=i) for i in range(3)])

da.save_binary('aux.bin', protocol='protobuf')
da_loaded = DocumentArray.load_binary('aux.bin', protocol='protobuf')

for doc in da_loaded:
    index = da.tags['id']
    print(animals[index])
```
````

## Rebuild Protobuf

To rebuild `docarray.proto` :

```bash
cd docarray
docker run -v $(pwd)/proto:/jina/proto jinaai/protogen
```