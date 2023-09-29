# Document

At the heart of `DocArray` lies the concept of [`BaseDoc`][docarray.base_doc.doc.BaseDoc].

A [BaseDoc][docarray.base_doc.doc.BaseDoc] is very similar to a [Pydantic](https://docs.pydantic.dev/)
[`BaseModel`](https://docs.Pydantic.dev/usage/models) -- in fact it _is_ a specialized Pydantic `BaseModel`. It allows you to define custom `Document` schemas (or `Model`s in
the Pydantic world) to represent your data.

!!! note
    Naming convention: When we refer to a `BaseDoc`, we refer to a class that inherits from [BaseDoc][docarray.base_doc.doc.BaseDoc]. 
    When we refer to a `Document` we refer to an instance of a `BaseDoc` class.

## Basic `Doc` usage

Before going into detail about what we can do with [BaseDoc][docarray.base_doc.doc.BaseDoc] and how to use it, let's
see what it looks like in practice.

The following Python code defines a `BannerDoc` class that can be used to represent the data of a website banner:

```python
from docarray import BaseDoc
from docarray.typing import ImageUrl


class BannerDoc(BaseDoc):
    image_url: ImageUrl
    title: str
    description: str
```

You can then instantiate a `BannerDoc` object and access its attributes:

```python
banner = BannerDoc(
    image_url='https://example.com/image.png',
    title='Hello World',
    description='This is a banner',
)

assert banner.image_url == 'https://example.com/image.png'
assert banner.title == 'Hello World'
assert banner.description == 'This is a banner'
```

## `BaseDoc` is a Pydantic `BaseModel`

The [BaseDoc][docarray.base_doc.doc.BaseDoc] class inherits from Pydantic [BaseModel](https://docs.pydantic.dev/usage/models). This means you can use
all the features of `BaseModel` in your `Doc` class. `BaseDoc`:

* Will perform data validation: `BaseDoc` will check that the data you pass to it is valid. If not, it will raise an 
error. Data being "valid" is actually defined by the type used in the type hint itself, but we will come back to this concept [later](../../data_types/first_steps.md).
* Can be configured using a nested `Config` class, see Pydantic [documentation](https://docs.pydantic.dev/usage/model_config/) for more detail on what kind of config Pydantic offers.
* Can be used as a drop-in replacement for `BaseModel` in your code and is compatible with tools that use Pydantic, like [FastAPI]('https://fastapi.tiangolo.com/').

## Representing multimodal and nested data

Let's say you want to represent a YouTube video in your application, perhaps to build a search system for YouTube videos.
A YouTube video is not only composed of a video, but also has a title, description, thumbnail (and more, but let's keep it simple).

All of these elements are from different [`modalities`](../../data_types/first_steps.md): the title and description are text, the thumbnail is an image, and the video itself is, well, a video.

DocArray lets you represent all of this multimodal data in a single object. 

Let's first create a `BaseDoc` for each of the elements that compose the YouTube video.

First for the thumbnail image:

```python
from docarray import BaseDoc
from docarray.typing import ImageUrl, ImageBytes


class ImageDoc(BaseDoc):
    url: ImageUrl
    bytes: ImageBytes = (
        None  # bytes are not always loaded in memory, so we make it optional
    )
```

Then for the video itself:

```python
from docarray import BaseDoc
from docarray.typing import VideoUrl, VideoBytes


class VideoDoc(BaseDoc):
    url: VideoUrl
    bytes: VideoBytes = (
        None  # bytes are not always loaded in memory, so we make it optional
    )
``` 

Then for the title and description (which are text) we'll just use a `str` type.

All the elements that compose a YouTube video are ready:

```python
from docarray import BaseDoc


class YouTubeVideoDoc(BaseDoc):
    title: str
    description: str
    thumbnail: ImageDoc
    video: VideoDoc
```

We now have `YouTubeVideoDoc` which is a pythonic representation of a YouTube video. 

This representation can be used to [send](../sending/first_step.md) or [store](../storing/first_step.md) data. You can even use it directly to [train a machine learning](../../how_to/multimodal_training_and_serving.md) [Pytorch](https://pytorch.org/docs/stable/index.html) model on this representation. 

!!! note

    You see here that `ImageDoc` and `VideoDoc` are also [BaseDoc][docarray.base_doc.doc.BaseDoc], and they are later used inside another [BaseDoc][docarray.base_doc.doc.BaseDoc]`.
    This is what we call nested data representation. 

    [BaseDoc][docarray.base_doc.doc.BaseDoc] can be nested to represent any kind of data hierarchy.

## Setting a Pydantic `Config` class

Documents support setting a custom `configuration` [like any other Pydantic `BaseModel`](https://docs.pydantic.dev/latest/api/config/).

Here is an example to extend the Config of a Document dependong on which version of Pydantic you are using.



=== "Pydantic v1"
    ```python
    from docarray import BaseDoc


    class MyDoc(BaseDoc):
        class Config(BaseDoc.Config):
            arbitrary_types_allowed = True  # just an example setting
    ```

=== "Pydantic v2"
    ```python
    from docarray import BaseDoc


    class MyDoc(BaseDoc):
        model_config = BaseDoc.ConfigDocArray.ConfigDict(
            arbitrary_types_allowed=True
        )  # just an example setting
    ```

See also:

* The [next part](./array.md) of the representing section
* API reference for the [BaseDoc][docarray.base_doc.doc.BaseDoc] class
* The [Storing](../storing/first_step.md) section on how to store your data 
* The [Sending](../sending/first_step.md) section on how to send your data

