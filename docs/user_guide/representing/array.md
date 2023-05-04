# Array of documents

DocArray allows users to represent and manipulate multi-modal data to build AI applications such as neural search and generative AI. 

As you have seen in the [previous section](array.md), the fundamental building block of DocArray is the [`BaseDoc`][docarray.base_doc.doc.BaseDoc] class which represents a *single* document, a *single* datapoint.

However, in machine learning we often need to work with an *array* of documents, and an *array* of data points.

This section introduces the concept of [`AnyDocArray`][docarray.array.AnyDocArray] which is an (abstract) collection of `BaseDoc`. This name of this library --
`DocArray` -- is derived from this concept and is short for `DocumentArray`.

## AnyDocArray

[`AnyDocArray`][docarray.array.AnyDocArray] is an abstract class that represents an array of [`BaseDoc`][docarray.BaseDoc]s which is not meant to be used directly, but to be subclassed.

We provide two concrete implementations of [`AnyDocArray`][docarray.array.AnyDocArray] :

- [`DocList`][docarray.array.doc_list.doc_list.DocList] which is a Python list of `BaseDoc`s
- [`DocVec`][docarray.array.doc_vec.doc_vec.DocVec] which is a column based representation of `BaseDoc`s

We will go into the difference between `DocList` and `DocVec` in the next section, but let's first focus on what they have in common.

The spirit of `AnyDocArray`s is to extend the `BaseDoc` and `BaseModel` concepts to the array level in a *seamless* way.

### Example

Before going into detail let's look at a code example.

!!! Note
    
    `DocList` and `DocVec` are both `AnyDocArray`s. The following section will use `DocList` as an example, but the same 
    applies to `DocVec`.

First you need to create a `Doc` class, our data schema. Let's say you want to represent a banner with an image, a title and a description:

```python
from docarray import BaseDoc, DocList
from docarray.typing import ImageUrl


class BannerDoc(BaseDoc):
    image: ImageUrl
    title: str
    description: str
```

Let's instantiate several `BannerDoc`s:

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

You can now collect them into a `DocList` of `BannerDoc`s:

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

`docs` here is an array-like collection of `BannerDoc`.

You can access documents inside it with the usual Python array API:

```python
print(docs[0])
```

```cmd
BannerDoc(image='https://example.com/image1.png', title='Hello World', description='This is a banner')
```

or iterate over it:

```python
for doc in docs:
    print(doc)
```

```cmd
BannerDoc(image='https://example.com/image1.png', title='Hello World', description='This is a banner')
BannerDoc(image='https://example.com/image2.png', title='Bye Bye World', description='This is (distopic) banner')
```

!!! note
    The syntax `DocList[BannerDoc]` might surprise you in this context.
    It is actually at the heart of DocArray, but we'll come back to it [later](#doclistdoctype-syntax) and continue with this example for now.

As we said earlier, `DocList` (or more generally `AnyDocArray`) extends the `BaseDoc` API at the array level.

What this means concretely is you can access your data at the Array level in just the same way you would access your data at the 
document level.

Let's see what that looks like:


At the document level:

```python
print(banner1.image)
```

```cmd
https://example.com/image1.png'
```

At the Array level:

```python
print(docs.image)
```

```cmd
['https://example.com/image1.png', 'https://example.com/image2.png']
```

!!! Important
    All the attributes of `BannerDoc` are accessible at the Array level.

!!! Warning
    Whereas this is true at runtime, static type analyzers like Mypy or IDEs like PyCharm will not be be aware of it.
    This limitation is known and will be fixed in the future by the introduction of plugins for Mypy, PyCharm and VSCode. 

This even works when you have a nested `BaseDoc`:

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
    content='Hello world is the most used example in programming, but do you know that? ...',
)

page2 = PageDoc(
    banner=BannerDoc(
        image='https://example.com/image2.png',
        title='Bye Bye World',
        description='This is (distopic) banner',
    ),
    content='What if the most used example in programming was Bye Bye World, would programming be that much fun? ...',
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

Yes, `docs.banner` returns a nested `DocList` of `BannerDoc`s! 

You can even access the attributes of the nested `BaseDoc` at the Array level:

```python
print(docs.banner.image)
```

```cmd
['https://example.com/image1.png', 'https://example.com/image2.png']
```

This is just the same way that you would do it with [BaseDoc][docarray.base_doc.doc.BaseDoc]:

```python
print(page1.banner.image)
```

```cmd
'https://example.com/image1.png'
```

### `DocList[DocType]` syntax

As you have seen in the previous section, `AnyDocArray` will expose the same attributes as the `BaseDoc`s it contains.

But this concept only works if (and only if) all of the `BaseDoc`s in the `AnyDocArray` have the same schema.

If one of your `BaseDoc`s has an attribute that the others don't, you will get an error if you try to access it at
the Array level.


!!! note
    To extend your schema to the Array level, `AnyDocArray` needs to contain a homogenous Document.

This is where the custom syntax `DocList[DocType]` comes into play.

!!! note
    `DocList[DocType]` creates a custom [`DocList`][docarray.array.doc_list.doc_list.DocList] that can only contain `DocType` Documents.

This syntax is inspired by more statically typed languages, and even though it might offend Python purists, we believe that it is a good user experience to think of an Array of `BaseDoc`s rather than just an array of non-homogenous `BaseDoc`s.

That said, `AnyDocArray` can also be used to create a non-homogenous `AnyDocArray`:

!!! note
    The default `DocList` can be used to create a non-homogenous list of `BaseDoc`.

!!! warning
    `DocVec` cannot store non-homogenous `BaseDoc` and always needs the `DocVec[DocType]` syntax.

The usage of a non-homogenous `DocList` is similar to a normal Python list but still offers DocArray functionality
like [serialization and sending over the wire](../sending/first_step.md). However, it won't be able to extend the API of your custom schema to the Array level.

Here is how you can instantiate a non-homogenous `DocList`:

```python
from docarray import BaseDoc, DocList
from docarray.typing import ImageUrl, AudioUrl


class ImageDoc(BaseDoc):
    url: ImageUrl


class AudioDoc(BaseDoc):
    url: AudioUrl


docs = DocList(
    [
        ImageDoc(url='https://example.com/image1.png'),
        AudioDoc(url='https://example.com/audio1.mp3'),
    ]
)
``` 

But this is not possible:

```python
try:
    docs = DocList[ImageDoc](
        [
            ImageDoc(url='https://example.com/image1.png'),
            AudioDoc(url='https://example.com/audio1.mp3'),
        ]
    )
except ValueError as e:
    print(e)
```

```cmd
ValueError: AudioDoc(
    id='e286b10f58533f48a0928460f0206441',
    url=AudioUrl('https://example.com/audio1.mp3', host_type='domain')
) is not a <class '__main__.ImageDoc'>
```

### `DocList` vs `DocVec`

[`DocList`][docarray.array.doc_list.doc_list.DocList] and [`DocVec`][docarray.array.doc_vec.doc_vec.DocVec] are both
[`AnyDocArray`][docarray.array.any_array.AnyDocArray] but they have different use cases, and differ in how
they store data in memory.

They share almost everything that has been said in the previous sections, but they have some conceptual differences.

[`DocList`][docarray.array.doc_list.doc_list.DocList] is based on Python Lists.
You can append, extend, insert, pop, and so on. In DocList, data is individually owned by each `BaseDoc` collect just
different Document references. Use [`DocList`][docarray.array.doc_list.doc_list.DocList] when you want to be able
to rearrange or re-rank your data. One flaw of `DocList` is that none of the data is contiguous in memory, so you cannot 
leverage functions that require contiguous data without first copying the data in a continuous array.

[`DocVec`][docarray.array.doc_vec.doc_vec.DocVec] is a columnar data structure. `DocVec` is always an array
of homogeneous Documents. The idea is that every attribute of the `BaseDoc` will be stored in a contiguous array: a column.

This means that when you access the attribute of a `BaseDoc` at the Array level, we don't collect the data under the hood 
from all the documents (like `DocList`) before giving it back to you. We just return the column that is stored in memory.

This really matters when you need to handle multimodal data that you will feed into an algorithm that requires contiguous data, like matrix multiplication
which is at the heart of Machine Learning, especially in Deep Learning.

Let's take an example to illustrate the difference:

Let's say you want to work with an Image:

```python
from docarray import BaseDoc
from docarray.typing import NdArray


class ImageDoc(BaseDoc):
    image: NdArray[
        3, 224, 224
    ] = None  # [3, 224, 224] this just mean we know in advance the shape of the tensor
``` 

And that you have a function that takes a contiguous array of images as input (like a deep learning model):

```python
def predict(image: NdArray['batch_size', 3, 224, 224]):
    ...
``` 

Let's create a `DocList` of `ImageDoc`s and pass it to the function:

```python
from docarray import DocList
import numpy as np

docs = DocList[ImageDoc](
    [ImageDoc(image=np.random.rand(3, 224, 224)) for _ in range(10)]
)

predict(np.stack(docs.image))
...
predict(np.stack(docs.image))
```

When you call `docs.image`, `DocList` loops over the ten documents and collects the image attribute of each document in a list. It is similar to doing:

```python
images = []
for doc in docs:
    images.append(doc.image)
```

this means that if you call `docs.image` multiple times, under the hood you will collect the image from each document and stack them several times. This is not optimal.

Let's see how it will work with `DocVec`:

```python
from docarray import DocList
import numpy as np

docs = DocList[ImageDoc](
    [ImageDoc(image=np.random.rand(3, 224, 224)) for _ in range(10)]
)

predict(docs.image)
...
predict(docs.image)
``` 

The first difference is that you don't need to call `np.stack` on `docs.image` because `docs.image` is already a contiguous array.
The second difference is that you just get the column and don't need to create it at each call.

One of the other main differences between both of them is how you can access documents inside them.

If you access a document inside a `DocList` you will get a `BaseDoc` instance, i.e. a document.

If you access a document inside a `DocVec` you will get a document view. A document view is a view of the columnar data structure which
looks and behaves like a `BaseDoc` instance. It is a `BaseDoc` instance but with a different way to access the data.

When you make a change at the view level it will be reflected at the DocVec level:

```python
from docarray import DocVec

docs = DocVec[ImageDoc](
    [ImageDoc(image=np.random.rand(3, 224, 224)) for _ in range(10)]
)

my_doc = docs[0]

assert my_doc.is_view()  # True
``` 

whereas with DocList:

```python
docs = DocList[ImageDoc](
    [ImageDoc(image=np.random.rand(3, 224, 224)) for _ in range(10)]
)

my_doc = docs[0]

assert not my_doc.is_view()  # False
```


!!! Note
    To summarize: you should use `DocVec` when you need to work with contiguous data, and you should use `DocList` when you need to rearrange
    or extend your data.


## Dealing with Optional Field


Both [`DocList`][docarray.array.doc_list.doc_list.DocList] and [`DocVec`][docarray.array.doc_vec.doc_vec.DocVec] support optional fields but they behave differently.

!!! note
    This whole section is specific for nested BaseDoc, DocList and DocVec. 
    for other data type DocList and DocVec will treat the optional case as a normal case.

Let's take an example:

```python
from typing import Optional
from docarray.typing import NdArray
import numpy as np


class ImageDoc(BaseDoc):
    tensor: NdArray


class ArticleDoc(BaseDoc):
    image: Optional[ImageDoc]
    title: str
```

In this example `ArticleDoc` has an optional field `image` which is an `ImageDoc`. This means that this field can either
be None or be a `ImageDoc` instance.

Remember for both DocList and DocVec calling `docs.image` will return a list like object of all the images of the documents.

For DocList it will just iterate over all the documents and collect the image attribute of each document in a sequence for DocVec it will return the column of the image attribute.

The question which kind of sequence to you pick when the field is optional, i.e, some of the datapoint could be None ?

For DocList it will return a list of `Optional[ImageDoc]` instead of a `DocList[ImageDoc]`:

```python
from docarray import DocList


docs = DocList[ArticleDoc](
    [
        ArticleDoc(image=ImageDoc(tensor=np.ones((3, 224, 224))), title="Hello"),
        ArticleDoc(image=None, title="World"),
    ]
)

assert docs.image == [ImageDoc(tensor=np.ones((3, 224, 224))), None]
```

but for DocVec it is a bit different. Indeed, DocVec store the data for each filed as contiguous column. 
This means that DocVec can create a column in only two case: either all the data for a field is None or all the data is not None.

For the first one the whole column will just be None. In the second The column will be a `DocList[ImageDoc]`



```python
from docarray import DocVec

docs = DocVec[ArticleDoc](
    [
        ArticleDoc(image=ImageDoc(tensor=np.zeros((3, 224, 224))), title="Hello")
        for _ in range(10)
    ]
)
assert (docs.image.tensor == np.zeros((3, 224, 224))).all()
``` 

Or it can be None

```python
docs = DocVec[ArticleDoc]([ArticleDoc(title="Hello") for _ in range(10)])
assert docs.image is None
``` 

But if you try a mix you will get an error:

```python
try:
    docs = DocVec[ArticleDoc](
        [
            ArticleDoc(image=ImageDoc(tensor=np.ones((3, 224, 224))), title="Hello"),
            ArticleDoc(image=None, title="World"),
        ]
    )
except ValueError as e:
    print(e)
``` 

```bash
None is not a <class '__main__.ImageDoc'>
```

--- 

See also:


* [First step](./first_step.md) of the representing section
* API Reference for the [`DocList`][docarray.array.doc_list.doc_list.DocList] class
* API Reference for the [`DocVec`][docarray.array.doc_vec.doc_vec.DocVec] class
* The [Storing](../storing/first_step.md) section on how to store your data 
* The [Sending](../sending/first_step.md) section on how to send your data
