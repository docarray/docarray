
# 🔤 Text

DocArray supports many different modalities including `Text`.
This section will show you how to load and handle text data using DocArray.

!!! tip
    Check out our predefined [`TextDoc`](#getting-started-predefined-textdoc) to get started and play around with our text features.

You can store text in DocArray like this:

```python
from docarray import BaseDoc


class MyText(BaseDoc):
    text: str = None


doc = MyText(text='Hello world!')
```

The text can include any type of characters, including emojis:

```python
doc.text = '👋	नमस्ते दुनिया!	你好世界！こんにちは世界！	Привет мир!'
```

## Load text file

If your text data is too long to be written inline or if it is stored in a file, you can also define the url as a [`TextUrl`][docarray.typing.url.text_url.TextUrl] first and then load the text data.

Let's first define a schema:

```python
from docarray import BaseDoc
from docarray.typing import TextUrl


class MyText(BaseDoc):
    text: str = None
    url: TextUrl = None
```
Next, you can instantiate a `MyText` object with a `url` attribute and load its content to the `text` field.
```python
doc = MyText(
    url='https://www.w3.org/History/19921103-hypertext/hypertext/README.html',
)
doc.text = doc.url.load()

assert doc.text.startswith('<TITLE>Read Me</TITLE>')
```

##  Segment long texts

Often times when you index or search text data, you don’t want to consider thousands of words as one huge string. 
Instead, some finer granularity would be nice. You can do this by leveraging nested fields. For example, let’s split some page content into its sentences by `'.'`.

```python
from docarray import BaseDoc, DocList


class Sentence(BaseDoc):
    text: str


class Page(BaseDoc):
    content: DocList[Sentence]


long_text = 'First sentence. Second sentence. And many many more sentences.'
page = Page(content=[Sentence(text=t) for t in long_text.split('.')])

page.summary()
```
<details>
    <summary>Output</summary>
    ``` { .text .no-copy }
    📄 Page : 13d909a ...
    └── 💠 content: DocList[Sentence]
        ├── 📄 Sentence : 6725382 ...
        │   ╭────────────────┬─────────────────────╮
        │   │ Attribute      │ Value               │
        │   ├────────────────┼─────────────────────┤
        │   │ text: str      │ First sentence      │
        │   ╰────────────────┴─────────────────────╯
        ├── 📄 Sentence : 17a934c ...
        │   ╭───────────────┬──────────────────────╮
        │   │ Attribute     │ Value                │
        │   ├───────────────┼──────────────────────┤
        │   │ text: str     │  Second sentence     │
        │   ╰───────────────┴──────────────────────╯
        └── ... 2 more Sentence documents
    ```
</details>

## Getting started - Predefined `TextDoc`

To get started and play around with your text data, DocArray provides a predefined [`TextDoc`][docarray.documents.text.TextDoc], which includes all of the previously mentioned functionalities:

``` { .python }
class TextDoc(BaseDoc):
    text: Optional[str]
    url: Optional[TextUrl]
    embedding: Optional[AnyEmbedding]
    bytes_: Optional[bytes]
```

