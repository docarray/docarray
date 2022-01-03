# Construct

````{tab} Empty document

```python
from jina import Document

d = Document()
```

````

````{tab} From attributes 

```python
from jina import Document
import numpy

d1 = Document(text='hello')
d2 = Document(buffer=b'\f1')
d3 = Document(blob=numpy.array([1, 2, 3]))
d4 = Document(uri='https://jina.ai',
             mime_type='text/plain',
             granularity=1,
             adjacency=3,
             tags={'foo': 'bar'})
```


```console
<jina.types.document.Document ('id', 'mime_type', 'text') at 4483297360>
<jina.types.document.Document ('id', 'buffer') at 5710817424>
<jina.types.document.Document ('id', 'blob') at 4483299536>
<jina.types.document.Document id=e01a53bc-aedb-11eb-88e6-1e008a366d48 uri=https://jina.ai mimeType=text/plain tags={'foo': 'bar'} granularity=1 adjacency=3 at 6317309200>
```

````


````{tab} From another Document

```python
from jina import Document

d = Document(content='hello, world!')
d1 = d

assert id(d) == id(d1)  # True
```

To make a deep copy, use `copy=True`:

```python
d1 = Document(d, copy=True)

assert id(d) == id(d1)  # False
```

````

`````{tab} From dict or JSON string

```python
from jina import Document
import json

d = {'id': 'hello123', 'content': 'world'}
d1 = Document(d)

d = json.dumps({'id': 'hello123', 'content': 'world'})
d2 = Document(d)
```

````{admonition} Parsing unrecognized fields
:class: tip

Unrecognized fields in a `dict`/JSON string are automatically put into the Document's `.tags` field:

```python
from jina import Document

d1 = Document({'id': 'hello123', 'foo': 'bar'})
```

```text
<jina.types.document.Document id=hello123 tags={'foo': 'bar'} at 6320791056>
```

You can use `field_resolver` to map external field names to `Document` attributes:

```python
from jina import Document

d1 = Document({'id': 'hello123', 'foo': 'bar'}, field_resolver={'foo': 'content'})
```

```text
<jina.types.document.Document id=hello123 mimeType=text/plain text=bar at 6246985488>
```

````


`````
