# Access Modality

```{tip}
It is strongly recommended to go through the {ref}`access-elements` section first before continuing.
```

Accessing modality means accessing the sub-Documents corresponding to a dataclass field. 

In the last chapter, we learned how to represent a multimodal document via `@dataclass` and type annotation from `docarray.typing`. We also learned that a multimodal dataclass can be converted into a `Document` object easily. That means if we have a list of multimodal dataclass objects, we can build a DocumentArray out of them:

```python
from docarray import Document, dataclass, DocumentArray
from docarray.typing import Image, Text


@dataclass
class MMDoc:
    banner: Image
    description: Text


da = DocumentArray(
    [
        Document(
            MMDoc(banner='test-1.jpeg', description='this is a test white-noise image')
        ),
        Document(
            MMDoc(banner='test-2.jpeg', description='another test image but in black')
        ),
    ]
)

da.summary()
```

```text
╭────────────── Documents Summary ───────────────╮
│                                                │
│   Length                    2                  │
│   Homogenous Documents      True               │
│   Has nested Documents in   ('chunks',)        │
│   Common Attributes         ('id', 'chunks')   │
│   Multimodal dataclass      True               │
│                                                │
╰────────────────────────────────────────────────╯
╭──────────────────────── Attributes Summary ────────────────────────╮
│                                                                    │
│   Attribute   Data type         #Unique values   Has empty value   │
│  ────────────────────────────────────────────────────────────────  │
│   chunks      ('ChunkArray',)   2                False             │
│   id          ('str',)          2                False             │
│                                                                    │
╰────────────────────────────────────────────────────────────────────╯
```

A natural question would be, how do we select those Documents that correspond to `MMDoc.banner`? 

This chapter describes how to select the sub-documents that correspond to a modality from a DocumentArray. So let me reiterate the logic here: when calling `Document()` to build Document object from a dataclass object, each field in that dataclass will generate a sub-document nested under `.chunks` or even `.chunks.chunks.chunks` at arbitrary level (except primitive types, which are stored in the `tags` of the root Document). To process a dataclass field via existing DocArray API/Jina/Hub Executor, we need a way to accurately select those sub-documents from the nested structure, which is the purpose of this chapter. 

## Selector Syntax

Following the syntax convention described in {ref}`access-elements`, a modality selector also starts with `@`, it uses `.` to indicate the field of the dataclass. Selecting a DocumentArray always results in another DocumentArray.

```text
@.[field1, field2, ...]
^^ ~~~~~~  ~~~~~~
||   |       |
||   |-------|
||       |
||       | --- indicate the field of dataclass
||
|| ------ indicate the start of modality selector
|
| ---- indicate the start of selector
```

Use the above DocumentArray as an example,

````{tab} Select Documents corresponding to .banner 

```python
da['@.[banner]']
```

```text
╭───────────────────────────── Documents Summary ──────────────────────────────╮
│                                                                              │
│   Length                 2                                                   │
│   Homogenous Documents   True                                                │
│   Common Attributes      ('id', 'parent_id', 'granularity', 'tensor',        │
│                          'mime_type', 'uri', 'modality')                     │
│   Multimodal dataclass   False                                               │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─────────────────────── Attributes Summary ────────────────────────╮
│                                                                   │
│   Attribute     Data type      #Unique values   Has empty value   │
│  ───────────────────────────────────────────────────────────────  │
│   granularity   ('int',)       1                False             │
│   id            ('str',)       2                False             │
│   mime_type     ('str',)       1                False             │
│   modality      ('str',)       1                False             │
│   parent_id     ('str',)       2                False             │
│   tensor        ('ndarray',)   2                False             │
│   uri           ('str',)       2                False             │
│                                                                   │
╰───────────────────────────────────────────────────────────────────╯
```


````

````{tab} Select Documents corresponding to .description 

```python
da['@.[description]']
```

```text
╭───────────────────────────── Documents Summary ──────────────────────────────╮
│                                                                              │
│   Length                 2                                                   │
│   Homogenous Documents   True                                                │
│   Common Attributes      ('id', 'parent_id', 'granularity', 'text',          │
│                          'modality')                                         │
│   Multimodal dataclass   False                                               │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
╭────────────────────── Attributes Summary ──────────────────────╮
│                                                                │
│   Attribute     Data type   #Unique values   Has empty value   │
│  ────────────────────────────────────────────────────────────  │
│   granularity   ('int',)    1                False             │
│   id            ('str',)    2                False             │
│   modality      ('str',)    1                False             │
│   parent_id     ('str',)    2                False             │
│   text          ('str',)    2                False             │
│                                                                │
╰────────────────────────────────────────────────────────────────╯
```


````

### Select multiple fields

You can select multiple fields by including them in the square brackets, separated by a comma `,`.

````{tab} Select Documents correspond to two fields

```python
da['@.[description, banner]']
```
````

````{tab} Result


```text
╭───────────────────────────── Documents Summary ──────────────────────────────╮
│                                                                              │
│   Length                        4                                            │
│   Homogenous Documents          False                                        │
│   2 Documents have attributes   ('id', 'parent_id', 'granularity', 'text',   │
│                                 'modality')                                  │
│   2 Documents have attributes   ('id', 'parent_id', 'granularity',           │
│                                 'tensor', 'mime_type', 'uri', 'modality')    │
│   Multimodal dataclass          False                                        │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
╭───────────────────────────── Attributes Summary ─────────────────────────────╮
│                                                                              │
│   Attribute     Data type                 #Unique values   Has empty value   │
│  ──────────────────────────────────────────────────────────────────────────  │
│   granularity   ('int',)                  1                False             │
│   id            ('str',)                  4                False             │
│   mime_type     ('str',)                  2                False             │
│   modality      ('str',)                  2                False             │
│   parent_id     ('str',)                  2                False             │
│   tensor        ('ndarray', 'NoneType')   4                True              │
│   text          ('str',)                  3                False             │
│   uri           ('str',)                  3                False             │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
```
````

### Slice dataclass objects

Remember each dataclass object corresponds to one Document object, you can first slice the DocumentArray before selecting the field. Specifically, you can do

```text
@r[slice].[field1, field2, ...]
```

where `slice` can be any slice syntax accepted in {ref}`access-elements`.

For example, to select the sub-Document `.banner` for only the first Document,

````{tab} Select .banner of the first dataclass  

```python
da['@r[:1].[banner]']
```

````

````{tab} Result

```text
╭───────────────────────────── Documents Summary ──────────────────────────────╮
│                                                                              │
│   Length                 1                                                   │
│   Homogenous Documents   True                                                │
│   Common Attributes      ('id', 'parent_id', 'granularity', 'tensor',        │
│                          'mime_type', 'uri', 'modality')                     │
│   Multimodal dataclass   False                                               │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─────────────────────── Attributes Summary ────────────────────────╮
│                                                                   │
│   Attribute     Data type      #Unique values   Has empty value   │
│  ───────────────────────────────────────────────────────────────  │
│   granularity   ('int',)       1                False             │
│   id            ('str',)       1                False             │
│   mime_type     ('str',)       1                False             │
│   modality      ('str',)       1                False             │
│   parent_id     ('str',)       1                False             │
│   tensor        ('ndarray',)   1                False             │
│   uri           ('str',)       1                False             │
│                                                                   │
╰───────────────────────────────────────────────────────────────────╯

```

````

### Slice `List[Type]` fields

If a field is annotated as a List of DocArray types, it will create a DocumentArray, one can add slicing after the field selector to further restrict the size of the sub-Documents.

```{code-block} python
---
emphasize-lines: 30
---
from typing import List

from docarray import Document, dataclass, DocumentArray
from docarray.typing import Image, Text


@dataclass
class MMDoc:
    banner: List[Image]
    description: Text


da = DocumentArray(
    [
        Document(
            MMDoc(
                banner=['test-1.jpeg', 'test-2.jpeg'],
                description='this is a test white image',
            )
        ),
        Document(
            MMDoc(
                banner=['test-1.jpeg', 'test-2.jpeg'],
                description='another test image but in black',
            )
        ),
    ]
)

for d in da['@.[banner][:1]']:
    print(d.uri)
```


```text
test-1.jpeg
test-1.jpeg
```

To summarize, slicing can be put in front of the field selector  to restrict the number of dataclass objects; or can be put after the field selector to restrict the number of sub-Documents.

### Select nested fields

A field can be annotated as a DocArray dataclass. In this case, the nested structure from the latter dataclass is copied to the former's `.chunks`. To select the deeply nested field, one can simply follow:

```text
@.[field1, field2, ...].[nested_field1, nested_field1, ...]
```

For example,

```{code-block} python
---
emphasize-lines: 23
---
from docarray import dataclass, Document, DocumentArray
from docarray.typing import Image, Text


@dataclass
class BannerDoc:
    description: Text = 'this is a test empty image'
    banner: Image = 'test-1.jpeg'


@dataclass
class ColumnArticle:
    featured: BannerDoc
    description: Text = 'this is a column article'
    website: str = 'https://jina.ai'


c1 = ColumnArticle(featured=BannerDoc(banner='test-1.jpeg'))
c2 = ColumnArticle(featured=BannerDoc(banner='test-2.jpeg'))

da = DocumentArray([Document(c1), Document(c2)])

for d in da['@.[featured].[banner]']:
    print(d.uri)
```

```text
test-1.jpeg
test-2.jpeg
```