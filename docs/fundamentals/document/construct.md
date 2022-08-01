(construct-doc)=
# Construct

Initializing a Document object is super easy. This chapter introduces the ways of constructing empty Document, filled Document. One can also construct Document from bytes, JSON, Protobuf message as introduced {ref}`in the next chapter<serialize>`.

## Construct an empty Document

```python
from docarray import Document

d = Document()
```

```text
<Document ('id',) at 5dd542406d3f11eca3241e008a366d49>
```

Every Document will have a unique random `id` that helps you identify this Document. It can be used to {ref}`access this Document inside a DocumentArray<access-elements>`.

````{tip}
The random `id` is the hex value of [UUID1](https://docs.python.org/3/library/uuid.html#uuid.uuid1). To convert it into the string of UUID:

```python
import uuid

str(uuid.UUID(d.id))
```
````

Though possible, it is not recommended modifying `.id` of a Document frequently, as this will lead to unexpected behavior.

(construct-from-dict)=
## Construct with attributes

This is the most common usage of the constructor: initializing a Document object with given attributes.

```python
from docarray import Document
import numpy

d1 = Document(text='hello')
d2 = Document(blob=b'\f1')
d3 = Document(tensor=numpy.array([1, 2, 3]))
d4 = Document(
    uri='https://jina.ai',
    mime_type='text/plain',
    granularity=1,
    adjacency=3,
    tags={'foo': 'bar'},
)
```

Don't forget to leverage autocomplete in your IDE.

```{figure} images/ide-autocomplete.png
:width: 80%
```

```text
<Document ('id', 'mime_type', 'text') at a14effee6d3e11ec8bde1e008a366d49>
<Document ('id', 'blob') at a14f00986d3e11ec8bde1e008a366d49> 
<Document ('id', 'tensor') at a14f01a66d3e11ec8bde1e008a366d49> 
<Document ('id', 'granularity', 'adjacency', 'mime_type', 'uri', 'tags') at a14f023c6d3e11ec8bde1e008a366d49>
```

````{tip}
When you `print()` a Document, you get a string representation such as `<Document ('id', 'tensor') at a14f01a66d3e11ec8bde1e008a366d49>`. It shows the non-empty attributes of that Document as well as its `id`, which helps you understand the content of that Document.

```text
<Document ('id', 'tensor') at a14f01a66d3e11ec8bde1e008a366d49>
          ^^^^^^^^^^^^^^    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                 |                          |
                 |                          |
          non-empty fields                  |
                                      Document.id
```
````



One can also wrap the keyword arguments into `dict`. The following ways of initialization have the same effect:

```python
d1 = Document(uri='https://jina.ai', mime_type='text/plain', granularity=1, adjacency=3)

d2 = Document(
    dict(uri='https://jina.ai', mime_type='text/plain', granularity=1, adjacency=3)
)

d3 = Document(
    {
        'uri': 'https://jina.ai',
        'mime_type': 'text/plain',
        'granularity': 1,
        'adjacency': 3,
    }
)
```

### Nested Document

```{seealso}
This section describes how to manually construct a nested Document, for example to hold different modalities, such as text and image.
\
To construct such multi-modal Documents in a more comfortabe, readable, and idiomatic way you should use DocArray's {ref}`dataclass <dataclass>` API.

To learn more about nested Document, please read {ref}`recursive-nested-document`.
```

Document can be nested inside `.chunks` and `.matches`. The nested structure can be specified directly during construction:

```python
from docarray import Document

d = Document(
    id='d0',
    chunks=[Document(id='d1', chunks=Document(id='d2'))],
    matches=[Document(id='d3')],
)

print(d)
```

```text
<Document ('id', 'chunks', 'matches') at d0>
```

For a nested Document, print its root does not give you much information. You can use {meth}`~docarray.document.mixins.plot.PlotMixin.summary`. For example, `d.summary()` gives you a more intuitive overview of the structure.

```text
 <Document ('id', 'chunks', 'matches') at d0>
    └─ matches
          └─ <Document ('id',) at d3>
    └─ chunks
          └─ <Document ('id', 'chunks') at d1>
              └─ chunks
                    └─ <Document ('id', 'parent_id', 'granularity') at d2>
```

When using in Jupyter notebook/Google Colab, Document is automatically prettified.

```{figure} images/doc-in-jupyter.png
```

(unk-attribute)=
### Unknown attributes handling

If you give an unknown attribute (i.e. not one of the built-in Document attributes), they will be automatically "caught" into `.tags` attributes. For example,

```python
from docarray import Document

d = Document(hello='world')

print(d, d.tags)
```

```text
<Document ('id', 'tags') at f957e84a6d4311ecbea21e008a366d49>
{'hello': 'world'}
```

You can change this "`catch`" behavior to `drop` (silently drop unknown attributes) or `raise` (raise a `AttributeError`) by specifying `unknown_fields_handler`. 

### Resolve unknown attributes with rules

One can resolve external fields into built-in attributes by specifying a mapping in `field_resolver`. For example, to resolve the field `hello` as the `id` attribute:

```python
from docarray import Document

d = Document(hello='world', field_resolver={'hello': 'id'})

print(d)
```

```text
<Document ('id',) at world>
```

One can see `id` of the Document object is set to `world`.


## Copy from another Document

To make a deep copy of a Document, use `copy=True`:

```python
from docarray import Document

d = Document(text='hello')
d1 = Document(d, copy=True)

print(d == d1, id(d) == id(d1))
```

```text
True False
```

That indicates `d` and `d1` have identical content, but they are different objects in memory.


If you want to keep the memory address of a Document object while only copying the content from another Document, you can use {meth}`~docarray.base.BaseDCType.copy_from`. 

```python
from docarray import Document

d1 = Document(text='hello')
d2 = Document(text='world')

print(id(d1))
d1.copy_from(d2)
print(d1.text)
print(id(d1))
```

```text
4479829968
world
4479829968
```

## What's next?

One can also construct Document from bytes, JSON, Protobuf message. These methods are introduced {ref}`in the next chapter<serialize>`.
