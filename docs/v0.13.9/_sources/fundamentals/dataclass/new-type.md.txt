# Support New Modality

Each type in `docarray.typing` corresponds to one modality. Supporting a new modality means adding a new type, and specifying how it is translated from/to Document.

Whether it is about adding a new type, or changing the behavior of an existing type, you can leverage the {meth}`~docarray.dataclasses.types.field` function.

## Create new types

Say you want to define a new type `MyImage`, where image is accepted as a URI, but instead of loading it to `.tensor` of the sub-document, you want to load it to `.blob`. This is different from the built-in `Image` type {ref}`behavior<type-list>`. 

All you need to do is:

```python
from docarray import Document

from typing import TypeVar

MyImage = TypeVar('MyImage', bound=str)


def my_setter(value) -> 'Document':
    return Document(uri=value).load_uri_to_blob()


def my_getter(doc: 'Document'):
    return doc.uri
```

Now you can use `MyImage` type in the dataclass:

````{tab} Use MyImage as type 
```python
from docarray import dataclass, field, Document


@dataclass
class MMDoc:
    banner: MyImage = field(setter=my_setter, getter=my_getter, default='test-1.jpeg')


Document(MMDoc()).summary()
```

````

````{tab} Document structure

```text
📄 Document: bde1ab74306c2f63188069879e3945ac
└── 💠 Chunks
    └── 📄 Document: cd594a6870a8921d7a9c6b0ec764251d
        ╭─────────────┬────────────────────────────────────────────────────────────────╮
        │ Attribute   │ Value                                                          │
        ├─────────────┼────────────────────────────────────────────────────────────────┤
        │ parent_id   │ bde1ab74306c2f63188069879e3945ac                               │
        │ granularity │ 1                                                              │
        │ blob        │ b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x0… │
        │             │ (length: 56810)                                                │
        │ mime_type   │ image/jpeg                                                     │
        │ uri         │ test-1.jpeg                                                    │
        ╰─────────────┴────────────────────────────────────────────────────────────────╯
```

````

Specifically, `setter` defines how you want to store the value in the sub-document. Usually you need to process it and fill the value into one of the attributes {ref}`defined by the Document schema<doc-fields>`. You may also want to keep the original value so that you can recover it in `getter` later. `setter` will be invoked when calling `Document()` on this dataclass.

`getter` defines how you want to recover the original value from the sub-Document. `getter` will be invoked when calling dataclass constructor given a Document object.

## Override existing types

To override `getter`, `setter` behavior of the existing types, you can define a map and pass it to the argument of `type_var_map` in the {meth}`~docarray.dataclasses.types.dataclass` function.

```python
from docarray import dataclass, field, Document
from docarray.typing import Image


def my_setter(value) -> 'Document':
    print('im setting .uri only not loading it!')
    return Document(uri=value)


def my_getter(doc: 'Document'):
    print('im returning .uri!')
    return doc.uri


@dataclass(
    type_var_map={
        Image: lambda x: field(setter=my_setter, getter=my_getter, _source_field=x)
    }
)
class MMDoc:
    banner: Image = field(setter=my_setter, getter=my_getter, default='test-1.jpeg')


m1 = MMDoc()
m2 = MMDoc(Document(m1))

assert m1 == m2
```

```text
im setting .uri only not loading it!
im returning .uri!
```