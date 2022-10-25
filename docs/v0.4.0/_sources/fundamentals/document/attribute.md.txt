# Access Attributes

Use `.` expression to get/set the value of an attribute as you would with any Python object: 

```python
from docarray import Document

d = Document()
d.text = 'hello world'

print(d.text)
```

```text
hello world
```

To unset attribute, simply assign it to `None`:

```python
d.text = None
```

or use {meth}`~docarray.base.BaseDCType.pop`:

```python
d.pop('text')
```


One can unset multiple attributes `.pop()`:

```python
d.pop('text', 'id', 'mime_type')
```

You can check which attributes are set by `.non_empty_fields`. 


## Content attributes

Among all attributes, content attributes, namely `.text`, `.tensor`, and `.blob` are super important as they contain the actual content.

They correspond to string-like data (e.g. for natural language), `ndarray`-like data (e.g. for image/audio/video data), and binary data for general purpose, respectively. 


| Attribute | Accept type                                                                                                                                                                            | Use case |
| --- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| --- |
| `doc.text` | Python string                                                                                                                                                                          | Contain text |
| `doc.tensor` | A Python (nested) list/tuple of numbers, Numpy `ndarray`, SciPy sparse matrix (`spmatrix`), TensorFlow dense & sparse tensor, PyTorch dense & sparse tensor, PaddlePaddle dense tensor | Contain image/video/audio |
| `doc.blob` | 	Binary string                                                                                                                                                                         | Contain intermediate IO buffer |


Each Document can contain only one type of content. That means these three attributes are mutually exclusive. Let's see an example:


```python
import numpy as np
from docarray import Document

d = Document(text='hello')
d.tensor = np.array([1, 2, 3])

print(d)
```

```text
<Document ('id', 'tensor', 'mime_type') at 7623808c6d6211ec9cf21e008a366d49>
```

As one can see `text` field is reset to empty.

But what if you want to represent more than one kind of information? Say, to fully represent a PDF page you need to store both image and text. In this case, you can use {ref}`nested Document<recursive-nested-document>`s by putting image into one sub-Document, and text into another sub-Document.

```python
from docarray import Document

d = Document(chunks=[Document(tensor=...), Document(text=...)])
```


The principle is each Document contains only one modality of information. In practice, this principle makes your full solution more clear and easier to maintain.

There is also a `.content` sugar getter/setter of the content fields. The content will be automatically grabbed or assigned to either `text`, `blob`, or `tensor` field based on the given type.

```python
from docarray import Document
d = Document(content='hello')
print(d)
```

```text
<Document ('id', 'mime_type', 'text') at b4d675466d6211ecae8d1e008a366d49>
```

```python
d.content = [1, 2, 3]
print(d)
```

```text
<Document ('id', 'tensor', 'mime_type') at 2808eeb86d6311ecaddb1e008a366d49>
```

You can also check which content field is set by `.content_type`.

(content-uri)=
## Load content from URI

A quite common pattern is loading content from a URI instead of assigning them directly in the code.

This can be easily done with `.uri` attribute. The value of `.uri` can point to either local URI, remote URI or [data URI](https://en.wikipedia.org/wiki/Data_URI_scheme).

````{tab} Local image URI


```python
from docarray import Document

d1 = Document(uri='apple.png').load_uri_to_image_tensor()
print(d1.content_type, d1.content)
```

```console
tensor [[[255 255 255]
  [255 255 255]
  [255 255 255]
  ...
```
````


````{tab} Remote text URI

```python
from docarray import Document

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
from docarray import Document

d1 = Document(uri='''data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA
AAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO
9TXL0Y4OHwAAAABJRU5ErkJggg==
''').load_uri_to_image_tensor()

print(d1.content_type, d1.content)
```
```console
tensor [[[255 255 255]
  [255   0   0]
  [255   0   0]
  [255   0   0]
  [255 255 255]]
  ...
```

````

There are more `.load_uri_to_*` functions that allow you to read {ref}`text<text-type>`, {ref}`image<image-type>`, {ref}`video<video-type>`, {ref}`3D mesh<mesh-type>`, {ref}`audio<audio-type>` and {ref}`tabular<table-type>` data. 

```{figure} images/doc-load-autocomplete.png
:width: 60%
```

```{admonition} Convert content to data URI
:class: tip
Inline data URI is helpful when you need a quick visualization in HTML, as it embeds all resources directly into that HTML. 

You can convert a URI to a data URI using `doc.load_uri_to_datauri()`. This will fetch the resource and make it inline.
```