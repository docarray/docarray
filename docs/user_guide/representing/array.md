# Collection of documents

DocArray allow users to represent and manipulate multi-modal data to build AI application (Generative AI, neural search ...). 
DocArray could be seen as a `multi-modal extension of Pydantic for Machine Learning use case`. 

!!! warning
    DocArray is actually more than just a Pydantic extension, it is a general purpose multi-modal python libraries. 
     But it is usefully to see it that way to fully understand the representing ability that DocArray offer.

As you have seen in the last section (LINK), the fundamental building block of DocArray is the [`BaseDoc`][docarray.base_doc.doc.BaseDoc] class which allows to represent a *single* document, a *single* datapoint.

In Machine Learning though we often need to work with a *collection* of documents, a *collection* of datapoints.

This section introduce the concept of `AnyDocArray` LINK which is an (abstract) collection of `BaseDoc`. This library
name: `DocArray` is actually derive from this concept, and it stands for `DocumentArray`.


## AnyDocArray

`AnyDocArray` is an abstract class that represent a collection of `BaseDoc` which is not meant to be used directly, but to be subclassed.

We provide two concrete implementation of `AnyDocArray` :

- [`DocList`][docarray.array.doc_list.doc_list.DocList] which is a python list of `BaseDoc`
- [`DocVec`][docarray.array.doc_vec.doc_vec.DocVec] which is a column based representation of `BaseDoc`

We will go into the difference between `DocList` and `DocVec` in the next section but let's first focus on what they have in common.


`AnyDocArray` spirit is to extend the `BaseDoc` and `BaseModel` concept to the Array level in a *seamless* way

!!! important
    `AnyDocArray` is the Array equivalent of a Pydantic `BaseModel`or a DocArray [`BaseDoc`][docarray.base_doc.doc.BaseDoc]. 
    It extends the `BaseDoc` API at the Array level.

### Example

before going into detail lets look at a code example. After all it all a question of API and code 
example is the best way to visualize an API.

!!! Note
    
    `DocList` and `DocVec` are both `AnyDocArray`. The following section will use `DocList` as an example, but the same 
    apply to `DocVec`.

First we need to create a Doc class, our data schema. Let's say we want to represent a banner with an image, a title and a description.

```python
from docarray import BaseDoc, DocList
from docarray.typing import ImageUrl


class BannerDoc(BaseDoc):
    image: ImageUrl
    title: str
    description: str
```

let's instantiate several `BannerDoc`

```python
banner1 = BannerDoc(
    image='https://example.com/image1.png',
    title='Hello World',
    description='This is a banner',
)

banner2 = BannerDoc(
    image='https://example.com/image2.png',
    title='Bye Bye World',
    description='This is (distopic) banner',
)
```

we can now collect them into a `DocList` of `BannerDoc`

```python
docs = DocList[BannerDoc]([banner1, banner2])

docs.summary()
```

```cmd
╭──────── DocList Summary ────────╮
│                                 │
│   Type     DocList[BannerDoc]   │
│   Length   2                    │
│                                 │
╰─────────────────────────────────╯
╭──── Document Schema ─────╮
│                          │
│   BannerDoc              │
│   ├── image: ImageUrl    │
│   ├── title: str         │
│   └── description: str   │
│                          │
╰──────────────────────────╯
```

`docs` here is a collection of `BannerDoc`. 

!!! note
    The syntax `DocList[BannerDoc]` should surprise you in this context,
    it is actually at the heart of DocArray but let's come back to it later LINK TO LATER and continue with the example.

As we said earlier `DocList` or more generaly `AnyDocArray` extend the `BaseDoc` API at the Array level.

What it means concretely is that the same way you can access with Pydantic at the attribute of your data at the 
document level, you can do access it  at the Array level.

Let's see how it looks:


at the document level:
```python
print(banner.url)
```

```cmd
https://example.com/image1.png'
```

at the Array level:
```python
print(docs.url)
```

```cmd
['https://example.com/image1.png', 'https://example.com/image2.png']
```

!!! Important
    All the attribute of `BannerDoc` are accessible at the Array level.

!!! Warning
    Whereas this is true at runtime, static type analyser like Mypy or IDE like PyCharm will not be able to know it.
    This limitation is know and will be fixed in the future by the introduction of a Mypy, PyCharm, VSCode plugin. 

This even work when you have a nested `BaseDoc`:

```python
from docarray import BaseDoc, DocList
from docarray.typing import ImageUrl


class BannerDoc(BaseDoc):
    image: ImageUrl
    title: str
    description: str


class PageDoc(BaseDoc):
    banner: BannerDoc
    content: str


page1 = PageDoc(
    banner=BannerDoc(
        image='https://example.com/image1.png',
        title='Hello World',
        description='This is a banner',
    ),
    content='Hello wolrd is the most used example in programming, but do you know that ? ...',
)

page2 = PageDoc(
    banner=BannerDoc(
        image='https://example.com/image2.png',
        title='Bye Bye World',
        description='This is (distopic) banner',
    ),
    content='What if the most used example in programming was Bye Bye World, would programming be that much fun ? ...',
)

docs = DocList[PageDoc]([page1, page2])

docs.summary()
```

```cmd
╭─────── DocList Summary ───────╮
│                               │
│   Type     DocList[PageDoc]   │
│   Length   2                  │
│                               │
╰───────────────────────────────╯
╭────── Document Schema ───────╮
│                              │
│   PageDoc                    │
│   ├── banner: BannerDoc      │
│   │   ├── image: ImageUrl    │
│   │   ├── title: str         │
│   │   └── description: str   │
│   └── content: str           │
│                              │
╰──────────────────────────────╯
``` 

```python
print(docs.banner)
``` 

```cmd
<DocList[BannerDoc] (length=2)>
```

Yes, `docs.banner` return a nested `DocList` of `BannerDoc` ! 

You can even access the attribute of the nested `BaseDoc` at the Array level:

```python
print(docs.banner.url)
```

```cmd
['https://example.com/image1.png', 'https://example.com/image2.png']
```

the same way that with Pydantic and DocArray [BaseDoc][docarray.base_doc.doc.BaseDoc] you would have done:

```python
print(page1.banner.image)
```

```cmd
'https://example.com/image1.png'
```

### Custom syntax and in depth understanding of `AnyDocArray`


