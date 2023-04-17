# 🗃 Multimodal

In this section, we will walk through how to use DocArray to process multiple data modalities in tandem. 

!!! tip "See also"
    In this section, we will work with image and text data. If you are not yet familiar with how to process these 
    modalities individually, you may want to check out the respective examples first: [`Image`](../image/image.md) 
    and [`Text`](../text/text.md)

## Model your data

If you work with multiple modalities at the same time, most likely they stand in some relation with each other. 
DocArray allows you to model your data and these relationships.

### Define a schema

Let's suppose you want to model a page of a newspaper that contains a main text, an image URL, a corresponding tensor 
as well as a description. You can model this example in the following way:

```python
from docarray import BaseDoc
from docarray.typing import ImageTorchTensor, ImageUrl


class Page(BaseDoc):
    main_text: str
    img_url: ImageUrl = None
    img_description: str = None
    img_tensor: ImageTorchTensor = None
```

### Instantiate an object

After extending [`BaseDoc`][docarray.BaseDoc] and defining your schema, you can instantiate an object with your actual 
data.

```python
page = Page(
    main_text='Hello world',
    img_url='https://github.com/docarray/docarray/blob/main/docs/assets/favicon.png?raw=true',
    img_description='This is the image of an apple',
)
page.img_tensor = page.img_url.load()

page.summary()
```
<details>
    <summary>Output</summary>
    ``` { .text .no-copy }
    📄 Page : 8f39674 ...
    ╭──────────────────────────────┬───────────────────────────────────────────────╮
    │ Attribute                    │ Value                                         │
    ├──────────────────────────────┼───────────────────────────────────────────────┤
    │ main_text: str               │ Hello world                                   │
    │ img_url: ImageUrl            │ https://github.com/docarray/docarray/blob/fe… │
    │                              │ ... (length: 90)                              │
    │ img_description: str         │ This is DocArray                              │
    │ img_tensor: ImageTorchTensor │ ImageTorchTensor of shape (320, 320, 3),      │
    │                              │ dtype: torch.uint8                            │
    ╰──────────────────────────────┴───────────────────────────────────────────────╯
    ```
</details>

### Access data 

After instantiation, each modality can be accessed directly from the `Page` object:

```python
print(page.main_text)
print(page.img_url)
print(page.img_description)
print(page.img_tensor)
```
<details>
    <summary>Output</summary>
    ``` { .text .no-copy }
    Hello world
    https://github.com/docarray/docarray/blob/main/docs/assets/favicon.png?raw=true
    This is DocArray
    ImageTorchTensor([[[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0],
                       ...,
                       [0, 0, 0]]])
    ```
</details>

### Nested data

If the data you want to model requires a more complex structure, nesting your attributes may be a good solution.

For this example, let's try to define a schema to represent a newspaper. The newspaper should consist of a cover page,
any number of following pages, and some metadata. Further, each page contains a main text and can contain an image 
and an image description.

To implement this you can simply add a `Newspaper` class to our previous implementation. The newspaper has a required 
`cover_page` attribute of type `Page` as well as a `pages` attribute, which is a `DocList` of `Page`s.

```python
from docarray import BaseDoc, DocList
from docarray.typing import ImageTorchTensor, ImageUrl


class Page(BaseDoc):
    main_text: str
    img_url: ImageUrl = None
    img_description: str = None
    img_tensor: ImageTorchTensor = None


class Newspaper(BaseDoc):
    cover: Page
    pages: DocList[Page] = None
    metadata: dict = None
```

You can instantiate this more complex `Newspaper` object in the same way as before:

```python
cover_page = Page(
    main_text='DocArray Daily',
    img_url='https://github.com/docarray/docarray/blob/main/docs/assets/favicon.png',
)

pages = DocList[Page](
    [
        Page(
            main_text='Hello world',
            img_url='https://github.com/docarray/docarray/blob/main/docs/assets/favicon.png',
            img_description='This is the image of an apple',
        ),
        Page(main_text='Second page'),
        Page(main_text='Third page'),
    ]
)

docarray_daily = Newspaper(
    cover=cover_page,
    pages=pages,
    metadata={'author': 'DocArray and friends', 'issue': '0.30.0'},
)

docarray_daily.summary()
```
<details>
    <summary>Output</summary>
    ``` { .text .no-copy }
    📄 Newspaper : 63189f7 ...
    ╭────────────────┬─────────────────────────────────────────────────────────────╮
    │ Attribute      │ Value                                                       │
    ├────────────────┼─────────────────────────────────────────────────────────────┤
    │ metadata: dict │ {'author': 'DocArray and friends', 'issue': '0.0.3 ... }    │
    │                │ (length: 2)                                                 │
    ╰────────────────┴─────────────────────────────────────────────────────────────╯
    ├── 🔶 cover: Page
    │   └── 📄 Page : ca164e3 ...
    │       ╭───────────────────┬──────────────────────────────────────────────────╮
    │       │ Attribute         │ Value                                            │
    │       ├───────────────────┼──────────────────────────────────────────────────┤
    │       │ main_text: str    │ DocArray Daily                                   │
    │       │ img_url: ImageUrl │ https://github.com/docarray/docarray/blob/feat-… │
    │       │                   │ ... (length: 81)                                 │
    │       ╰───────────────────┴──────────────────────────────────────────────────╯
    └── 💠 pages: DocList[Page]
        ├── 📄 Page : 64ed19c ...
        │   ╭──────────────────────┬───────────────────────────────────────────────╮
        │   │ Attribute            │ Value                                         │
        │   ├──────────────────────┼───────────────────────────────────────────────┤
        │   │ main_text: str       │ Hello world                                   │
        │   │ img_url: ImageUrl    │ https://github.com/docarray/docarray/blob/fe… │
        │   │                      │ ... (length: 81)                              │
        │   │ img_description: str │ DocArray logoooo                              │
        │   ╰──────────────────────┴───────────────────────────────────────────────╯
        ├── 📄 Page : 4bd7e45 ...
        │   ╭─────────────────────┬────────────────╮
        │   │ Attribute           │ Value          │
        │   ├─────────────────────┼────────────────┤
        │   │ main_text: str      │ Second page    │
        │   ╰─────────────────────┴────────────────╯
        └── ... 1 more Page documents
    ```
</details>