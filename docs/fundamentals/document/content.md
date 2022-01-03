# Content 

{attr}`~docarray.Document.text`, {attr}`~docarray.Document.blob`, and {attr}`~docarray.Document.buffer` are the three content attributes of a Document. They correspond to string-like data (e.g. for natural language), `ndarray`-like data (e.g. for image/audio/video data), and binary data for general purpose, respectively. Each Document can contain only one type of content.

| Attribute | Accept type                                                                                                                                                                            | Use case |
| --- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| --- |
| `doc.text` | Python string                                                                                                                                                                          | Contain text |
| `doc.blob` | A Python (nested) list/tuple of numbers, Numpy `ndarray`, SciPy sparse matrix (`spmatrix`), TensorFlow dense & sparse tensor, PyTorch dense & sparse tensor, PaddlePaddle dense tensor | Contain image/video/audio |
| `doc.buffer` | 	Binary string                                                                                                                                                                         | Contain intermediate IO buffer |

````{admonition} Exclusivity of the content
:class: important

Note that one `Document` can only contain one type of `content`: either `text`, `buffer`, or `blob`. If you set one, the others will be cleared. 

```python
import numpy as np

d = Document(text='hello')
d.blob = np.array([1])

d.text  # <- now it's empty
```

````

````{admonition} Why a Document contains only data type
:class: question

What if you want to represent more than one kind of information? Say, to fully represent a PDF page you need to store both image and text. In this case, you can use {ref}`nested Document<recursive-nested-document>`s by putting image into one sub-Document, and text into another.

```python
d = Document(chunks=[Document(blob=...), Document(text=...)])
```


The principle is each Document contains only one modality. This makes the whole logic clearer.
````

```{tip}
There is also a `doc.content` sugar getter/setter of the above non-empty field. The content will be automatically grabbed or assigned to either `text`, `buffer`, or `blob` field based on the given type.
```



## Load content from URI

Often, you need to load data from a URI instead of assigning them directly in your code, {attr}`~docarray.Document.uri` is the attribute you must learn. 

After setting `.uri`, you can load data into `.text`/`.buffer`/`.blob` as follows.

The value of `.uri` can point to either local URI, remote URI or [data URI](https://en.wikipedia.org/wiki/Data_URI_scheme).

````{tab} Local image URI


```python
from jina import Document

d1 = Document(uri='apple.png').load_uri_to_image_blob()
print(d1.content_type, d1.content)
```

```console
blob [[[255 255 255]
  [255 255 255]
  [255 255 255]
  ...
```
````


````{tab} Remote text URI

```python
from jina import Document

d1 = Document(uri='https://www.gutenberg.org/files/1342/1342-0.txt').load_uri_to_text()

print(d1.content_type, d1.content)
```


```console
text ﻿The Project Gutenberg eBook of Pride and Prejudice, by Jane Austen

This eBook is for the use of anyone anywhere in the United States and
most other parts of the wor
```
````

````{tab} Inline data URI

```python
from jina import Document

d1 = Document(uri='''data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA
AAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO
9TXL0Y4OHwAAAABJRU5ErkJggg==
''').load_uri_to_image_blob()

print(d1.content_type, d1.content)
```
```console
blob [[[255 255 255]
  [255   0   0]
  [255   0   0]
  [255   0   0]
  [255 255 255]]
  ...
```

````

There are more `.load_uri_to_*` functions that allow you to read {ref}`text<text-type>`, {ref}`image<image-type>`, {ref}`video<video-type>`, {ref}`3D mesh<mesh-type>`, {ref}`audio<audio-type>` and {ref}`tabular<table-type>` data into Jina.

```{admonition} Write to data URI
:class: tip
Inline data URI is helpful when you need a quick visualization in HTML, as it embeds all resources directly into that HTML. 

You can convert a URI to a data URI using `doc.load_uri_to_datauri()`. This will fetch the resource and make it inline.
```
