# Construct

Just like Python dataclasses module, DocArray provides a decorator {meth}`~docarray.dataclasses.types.dataclass` and a set of type annotations such as `Image`, `Text`, `Audio`, that allow you to construct multimodal Document in the following way.

```python
from docarray import dataclass, Image, Text


@dataclass
class MyMutilModalDoc:
    avatar: Image
    description: Text


m = MyMutilModalDoc(avatar='test-1.jpeg', description='hello, world')
```

To convert it into a `Document` object, simply:

```python
from docarray import Document

d = Document(m)
d.summary()
```

One can see that this creates a Document object with two chunks nested.

```text
📄 Document: f3b193bbe8403c3ce1599b82f941f68a
└── 💠 Chunks
    ├── 📄 Document: 18c7ca1c829fe819250faa4914bc45c1
    │   ╭──────────────┬───────────────────────────────────────────────────────────────╮
    │   │ Attribute    │ Value                                                         │
    │   ├──────────────┼───────────────────────────────────────────────────────────────┤
    │   │ parent_id    │ f3b193bbe8403c3ce1599b82f941f68a                              │
    │   │ granularity  │ 1                                                             │
    │   │ tensor       │ <class 'numpy.ndarray'> in shape (504, 504, 3), dtype: uint8  │
    │   │ mime_type    │ image/jpeg                                                    │
    │   │ uri          │ test-1.jpeg                                                   │
    │   │ modality     │ image                                                         │
    │   ╰──────────────┴───────────────────────────────────────────────────────────────╯
    └── 📄 Document: 1ee7fadddc23fc72365b2069f82d4bb4
        ╭──────────────────────┬───────────────────────────────────────────────────────╮
        │ Attribute            │ Value                                                 │
        ├──────────────────────┼───────────────────────────────────────────────────────┤
        │ parent_id            │ f3b193bbe8403c3ce1599b82f941f68a                      │
        │ granularity          │ 1                                                     │
        │ text                 │ hello, world                                          │
        │ modality             │ text                                                  │
        ╰──────────────────────┴───────────────────────────────────────────────────────╯
```

To convert a Document object back to a `MyMutilModalDoc` object,

```python
m = MyMutilModalDoc(d)
```


## Dataclass decorator

First, you need to import `dataclass` decorator from DocArray package:

```python
from docarray import dataclass
```

The look and feel of this `dataclass` decorator is the same as the built-in `dataclass` decorator. In fact, any class wrapped by `docarray.dataclass` is also a valid Python `dataclass`:

```{code-block} python
---
emphasize-lines: 3, 6
---
from dataclasses import is_dataclass

from docarray import dataclass, Image


@dataclass
class MMDoc:
    banner: Image = 'test-1.jpeg'


print(is_dataclass(MMDoc))
print(is_dataclass(MMDoc()))
```


```text
True
True
```

That means, [arguments accepted by standard `dataclass`](https://docs.python.org/3/library/dataclasses.html#dataclasses.dataclass) are also accepted here. Methods that can be applied to Python `dataclass` can be also applied to DocArray `dataclass`.

To tell if a class or object is DocArray's dataclass, you can use {meth}`~docarray.dataclasses.types.is_multimodal`:

```python
import dataclasses

import docarray
from docarray.dataclasses.types import is_multimodal


@docarray.dataclass
class MMDoc1:
    banner: docarray.Image = 'test-1.jpeg'


@dataclasses.dataclass
class MMDoc2:
    banner: docarray.Image = 'test-1.jpeg'


print(is_multimodal(MMDoc1))
print(is_multimodal(MMDoc2))
```

```text
True
False
```


## Annotate class fields

One can annotate a class field as `Image`, `Text`, `JSON`, `Audio`, or as primitive Python types, or as other `docarray.dataclass`. For example,

```python
from docarray import dataclass, Image, Text, Audio


@dataclass
class MMDoc2:
    banner: Image = 'test-1.jpeg'
    summary: Text = 'This is an empty test image in 256 x 256 pixels'
    soundfx: Audio = 'white-noise.wav'
```

Convert `MMDoc2` object into a `Document` object is easy, simply via
```python
from docarray import Document

m = MMDoc2()
d = Document(m)
```

One can look at the structure of `d` via `d.summary()`:


```text
📄 Document: 90c744c5155c2356d27f8c91955f70f7
└── 💠 Chunks
    ├── 📄 Document: c9d71990088fb0d8db3c83a6bd35650d
    │   ╭──────────────┬───────────────────────────────────────────────────────────────╮
    │   │ Attribute    │ Value                                                         │
    │   ├──────────────┼───────────────────────────────────────────────────────────────┤
    │   │ parent_id    │ 90c744c5155c2356d27f8c91955f70f7                              │
    │   │ granularity  │ 1                                                             │
    │   │ tensor       │ <class 'numpy.ndarray'> in shape (504, 504, 3), dtype: uint8  │
    │   │ mime_type    │ image/jpeg                                                    │
    │   │ uri          │ test-1.jpeg                                                   │
    │   │ modality     │ image                                                         │
    │   ╰──────────────┴───────────────────────────────────────────────────────────────╯
    ├── 📄 Document: 22e27708288e70813673711c86f834ee
    │   ╭─────────────────┬────────────────────────────────────────────────────────────╮
    │   │ Attribute       │ Value                                                      │
    │   ├─────────────────┼────────────────────────────────────────────────────────────┤
    │   │ parent_id       │ 90c744c5155c2356d27f8c91955f70f7                           │
    │   │ granularity     │ 1                                                          │
    │   │ text            │ This is an empty test image in 256 x 256 pixels            │
    │   │ modality        │ text                                                       │
    │   ╰─────────────────┴────────────────────────────────────────────────────────────╯
    └── 📄 Document: 05ad36dfb0c520027b18c582d205c176
        ╭──────────────┬───────────────────────────────────────────────────────────────╮
        │ Attribute    │ Value                                                         │
        ├──────────────┼───────────────────────────────────────────────────────────────┤
        │ parent_id    │ 90c744c5155c2356d27f8c91955f70f7                              │
        │ granularity  │ 1                                                             │
        │ tensor       │ <class 'numpy.ndarray'> in shape (63248,), dtype: float32     │
        │ modality     │ audio                                                         │
        ╰──────────────┴───────────────────────────────────────────────────────────────╯
```

### Understand the consequence of field annotation

One DocArray `dataclass` corresponds to one `Document` object, let's call it `root`.

- Everytime a class field is annotated as a DocArray type, a nested sub-Document under `root.chunks`.
- The annotation type determines how the sub-Document is constructed. For example, annotating a field as `Image` will instruct the construction to fill in `doc.tensor` by reading the image URI. Annotating a field as `JSON` will instruct the construction to fill in `doc.tags`. The complete behavior table can be found below:

| Type annotation | Accepted value types                   | Behavior                                                                                     |
|-----------------|----------------------------------------|----------------------------------------------------------------------------------------------|
| `Image`         | `str`, `PIL.image`, tensor-like object | Create a sub-Document, fill in `doc.tensor` by reading the image and set `.modality='image'` |
| `Text`          | `str`                                  | Create a sub-Document, fill in `doc.text` by the value and set `.modality='text'`            |
| `Audio`         | `str`                                  | Create a sub-Document, fill in `doc.tensor` and `.modality='text'`                           |
| `JSON`          | `str`, `Dict`                          | Create a sub-Document, fill in `doc.tags`                                                    |

