# Set/unset Attributes

Set an attribute as you would with any Python object: 

```python
from docarray import Document

d = Document()
d.text = 'hello world'
```

```text
<jina.types.document.Document id=9badabb6-b9e9-11eb-993c-1e008a366d49 mime_type=text/plain text=hello world at 4444621648>
```


To unset attribute, simply assign it to `None`:

```python
d.text = None
```

or use {meth}`~docarray.Document.pop`:

```python
d.pop('text')
```

```text
<jina.types.document.Document id=cdf1dea8-b9e9-11eb-8fd8-1e008a366d49 mime_type=text/plain at 4490447504>
```


One can unset multiple attributes with {meth}`~docarray.Document.pop`:

```python
d.pop('text', 'id', 'mime_type')
```

```text
<jina.types.document.Document at 5668344144>
```

## Tags

`Document` contains the {attr}`~docarray.Document.tags` attribute that can hold a map-like structure that can map arbitrary values. 
In practice, you can store meta information in `tags`.

```python
from jina import Document

doc = Document(tags={'dimensions': {'height': 5.0, 'weight': 10.0, 'last_modified': 'Monday'}})

doc.tags['dimensions']
```

```text
{'weight': 10.0, 'height': 5.0, 'last_modified': 'Monday'}
```