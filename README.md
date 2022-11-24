# DocArray v2

this repo is a PoC for the new version of DocArray. The scope of the PoC is twofolds:

* Mininal pydantic like API to feel/grasp the new user interface
* Protobuf serialization/deserialization


the key ideas for this new version of DocArray:

* rely on pydantic as much as possible
* More abstract and powerful concept with predefined easy to use object
* explicit better than implicit. (We can't afford implicit with a higher level of abstraction )

## Document schema API

DocArray v2 is based on pydantic schema. A Document is nothing more than a Pydantic Model with a predefined Id field and a protobuf support. We provide predefined Document for different modality

```python
from docarray import Document

doc = Document()
doc
```




    BaseDocument(id=UUID('88819868-54fe-4316-9bf1-4af650e0e631'))



To extend a Document you need to extend the schema by creating a new class inheriting Document.

This follow [Pydantic Model](https://pydantic-docs.helpmanual.io/usage/models/) API.

It is similar to the dataclass from the (old) docarray


```python
from docarray.typing import Tensor
import numpy as np


class Banner(Document):
    text: str
    image: Tensor


banner = Banner(text='DocArray is amazing', image=np.zeros((3, 224, 224)))
banner
```




    Banner(id=UUID('873604ed-c164-4fe5-8427-9b4e698b8dfb'), text='DocArray is amazing', image=array([[[0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            ...,

            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.]]]))



Note: there is no pretty print (from rich) but it is just a PoC

You can represent nester document as well 


```python
class NestedDocument(Document):
    title: str
    banner: Banner


doc = NestedDocument(title='Jina is amazing', banner=banner)
doc
```




    NestedDocument(id=UUID('56ca3ae1-ca71-4258-a760-af9e09dea93f'), title='Jina is amazing', banner=Banner(id=UUID('873604ed-c164-4fe5-8427-9b4e698b8dfb'), text='DocArray is amazing', image=array([[[0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            ...,
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.]]])))



### Inheritance and composition

Before we showed how to **compose** Document. You can as well **extend** Document by inheritance


```python
class ExtendNestedDocument(NestedDocument):
    warning: str


extended_doc = ExtendNestedDocument(
    title='Jina is amazing', banner=banner, warning='hello'
)
extended_doc
```




    ExtendNestedDocument(id=UUID('1a973e74-7774-4387-9b26-e5b5c8a0cfe4'), title='Jina is amazing', banner=Banner(id=UUID('873604ed-c164-4fe5-8427-9b4e698b8dfb'), text='DocArray is amazing', image=array([[[0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            ...,

            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.]]])), warning='hello')




### Predefined Document

A Document has only ID has a predefined field, no more text, uri, tensor embedding. This is the user that need to construct their abstraction. Nevertheless we don't want to loose the handiness of the old Document that provide predifined fields. Therefore we provide predfined mono modals building blocks that are just predifined Document that cover common use case the same way the old Documennt was doing.

This is just an example and the real predefined one need to think in depth


```python
from docarray import Text, Image

doc_text = Text(text='hello')
doc_text
```




    Text(id=UUID('c8249d8d-473c-47e1-84fd-961aba81d74e'), text='hello', tensor=None)




```python
doc_image = Image(uri='http://jina.ai')
doc_image
```




    Image(id=UUID('a461e508-4c8b-430b-a63f-091007743cf3'), uri=ImageUrl('http://jina.ai', scheme='http', host='jina.ai', tld='ai', host_type='domain'), tensor=None)



### What about helper function ?

The old way of using helper function (`doc.load_uri_to_image_tensor()`) does not work anymore because we can't operate at a Document levels since we don't know what field are in the Document. A better approach is to encode any kind of modality helper in the type directly. The assignment is then explicit, we cannot afford implicit because we technically can have multi field (embedding, tensor). S


```python
doc_image.tensor = doc_image.uri.load()
doc_image
```




    Image(id=UUID('a461e508-4c8b-430b-a63f-091007743cf3'), uri=ImageUrl('http://jina.ai', scheme='http', host='jina.ai', tld='ai', host_type='domain'), tensor=array([[[0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            ...,

            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.]]]))

## Example: working with Embedding

All of the predefined Document have a predefined Embedding field


```python
image = Image(embedding=np.zeros((100, 1)))
assert image.embedding is not None
```

We can easily extend them to have multi embedding:



```python
from typing import Optional
from docarray.typing import Embedding


class MyExtendedImage(Image):
    embedding2: Embedding
    embedding3: Optional[Embedding]
```


```python
image = MyExtendedImage(embedding=np.zeros((100, 1)), embedding2=np.zeros((100, 1)))
assert image.embedding is not None
assert image.embedding2 is not None
assert image.embedding3 is None
```

Atm we have a couple of method that work on embedding:
* embed
* find/match
...

They all work on the `embedding` field. Nevertheless we don't have nesceraly this field define. User can have Document without embedding, or have embedding that are not call embedding, or have multiple embeddings ...

The solution is the same as for Executor ( see below ). We use pick automatically an the first embedding field by default and we allow users to explicitly define the mapping if they want to


```python
# THIS CODE DOES NOT RUN YET

da.find(da2, 'embedding:embedding1')
```

## DocumentArray

a DocumentArray is a list like container of Document. The big change with the (old) DocArray is that now DocumentArray can precise on Document Schema on which they work on. This is usefull both for type hint and for protobuf reconstruction.

The old behavior where DocumentArray could contain any kind of Document is still possible (it is actually the default) because we have the a Schemaless Document.

```python
from docarray import Document, DocumentArray, Text, Image

da = DocumentArray([Text(text='hello'), Image(tensor=np.zeros((3, 224, 224)))])
da
```




    [Text(id=UUID('b901bd1d-cd4d-4f17-b774-e6ddb8487234'), text='hello', tensor=None),
     Image(id=UUID('1849bdb9-d1da-4b18-8670-84f1af053693'), uri=None, tensor=array([[[0., 0., 0., ..., 0., 0., 0.],
             [0., 0., 0., ..., 0., 0., 0.],
             [0., 0., 0., ..., 0., 0., 0.],
             ...,

             [0., 0., 0., ..., 0., 0., 0.],
             [0., 0., 0., ..., 0., 0., 0.],
             [0., 0., 0., ..., 0., 0., 0.]]]))]



inside the DocumentArray there is a typed define


```python
da.document_type
```




    docarray.document.any_document.AnyDocument



in this case the schema of the DocumentArray is the AnyDocument schema, i.e, it works with any Document

but you can as well restraint this type


```python
da = DocumentArray[Text]([Text(text='hello'), Text(text='bye bye')])
da
```




    [Text(id=UUID('82d422a6-2805-4d70-8f5b-cf6a8e720e3c'), text='hello', tensor=None),
     Text(id=UUID('1a624b5e-fc7d-46b8-916e-4f5467e8cf82'), text='bye bye', tensor=None)]



Note this is a experiment API, we can rely on DocumentArray(..., type=Text) in a metaclass way (like storage) otherwise. It is a proposition I like it this way better


```python
da.document_type
```




    docarray.predefined_document.text.Text



This is mainly usefull for type hint:


```python
def do_smth_on_da(da: DocumentArray[Text]):

    for doc in da:
        print(
            da.text
        )  ## this will work since you expect Document Text inside the DocumentArray
```

### Nested DocumentArray inside Document

(Old) Document had the chunks field to represented nested document in document. We extend this principle by allowing a field of Document to just be a DocumentArray


```python
from docarray import Document, Image


class Video(Document):
    title: str
    frames: DocumentArray[Image]


frames = DocumentArray([Image(tensor=np.zeros((3, 224, 224))) for _ in range(24)])
video = Video(title='hello', frames=frames)
video
```




    Video(id=UUID('4d3f3d99-a346-4152-bd81-9170136ca75b'), title='hello', frames=[Image(id=UUID('69f2c62f-bdec-4172-aaef-630029e3e974'), uri=None, tensor=array([[[0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],


            ...,
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.]]])), Image(id=UUID('dfc9c251-5a79-4278-8748-defa18a5bd65'), uri=None, tensor=array([[[0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            ...,
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.]],
    
           
           [[0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            ...,
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.]]])), Image(id=UUID('1e9027ce-e036-4bc2-85b2-e2f97c3a1200'), uri=None, tensor=array([[[0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            ...,
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.]]]))])



## Jina side : Executor interoperability

The API above is much more flexible than the current Document implementation. This buys us better multimodal support as well as more natural vector DB integration. On the other hand, the Executor have less structure to rely on.
We intend to tackle this limitation in the following way:

- **Every Executor expects a schema that it will work on**. This could be self-defined, or an imported ‘default’ Document:

```python
class MyExecSchema(Document):
    text: str
    embedding: Embedding


class MyExec(Executor):
    @requests
    def foo(docs: DocumentArray[MyExecSchema], *args, **kwargs):
        ...
```

- On the client side, the user can (**but does not have to!)** define a translation (`schema_map`, name not final) from their schema to the expected schema:

```python
class ClientDoc(Document):
    text: str
    first_embedding: Embedding
    second_embedding: Embedding


doc = ClientDoc(...)
# map `text` to `text` and `first_embedding` to `embedding`
client.post(doc, schema_map={'MyExec': 'text:text,first_embedding:embedding'})
# the case of nested schema can be handle with dunder notation
```

- The **worker runtime performs the schema translation**, by simply renaming fields in the received document. That way the Executor only gets to see what it wants. We plan on doing automatic translation as well to avoid verbosity when it is not needed (see below)
- **Defaults**: If the client schema already matches the Executor schema, no translation is necessary and no schema map needs to be passed.
If the schemas do not match and no map is provided, the runtime can do a best effort translation based on the types. We clearly document that mechanism and avoid any surprises.
- **Backward compatibility:** If an Executor fails to provide an expected schema, we assume the current (legacy) schema that already exists for every Document
- Executors can receive any schema that is superset of his defined schema. Clients send to the Flow, documents with a schema that is valid to all Executors. 
In the worker runtime, deserializing docs from a python object to protobuf should rely on the initial protobuf message, rather [than creating a new one](https://github.com/jina-ai/docarray/blob/main/docarray/proto/io/__init__.py#L41)

**Automatic translation details:** 

Lets say my input data follow this schema

```python
class ImageTextDocument(Document):
    text: Text
    image: Image
    embedding: Embedding
```

and that my Executor follow this one

```python
class MyPhoto(Document):
    vector: Embedding
    photo: Image
    description: Text


class PhotoEmbeddingExecutor(Executor):
    @requests
    def encode(self, docs: DocumentArray[MyPhoto], **kwargs):
        for doc_ in docs:
            doc.embedding = self.image_model(doc.photo)
```

They define actually the same underlying schema but with different field name. So the way would be to do

```python
client.post(doc, schema_map={'MyExec': 'image:photo,embedding:vector,text:description'})
```

But this is too verbose for smth just translating the same schema. We will do that automatically., How ? We look at the field type and do a one by one group by.

What if the matching is not exact ? i.e what if we have the two following schema ?

```python
class ClientDoc(Document):
    text: Text
    embedding1: Embedding
    embedding2: Embedding


class ExecutorDoc(Document):
    text: Text
    embedding: Embedding
```

 If we have collision on a field (here two embeddings) we will take the first field that correspond (in this case embedding1). This is a deterministic algorithm because fields are ordered in pydantic

