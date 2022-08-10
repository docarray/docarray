(subindex)=
# Subindices

Subindices allow you to efficiently search through multimodal or nested Documents.

```{seealso}
To see an example of subindices in action, see {ref}`here <multimodal-search-example>`.
```

Each subindex indexes and stores one nesting level, such as `'@c'` or a {ref}`custom modality <dataclass>` like `'@.[image]'`, and makes it directly searchable.
Under the hood, subindices are fully fledged DocumentArrays with their own Document Store.


```{admonition} Document stores
:class: hint

Subindices are available for all DocumentArray types, including in-memory, but are most useful in combination with a {ref}`Document Store <doc-store>`.
```

## Construct subindices

Subindices are constructed when creating a DocumentArray,
by passing a configuration for each desired subindex to the `subindex_configs` parameter:

````{tab} Subindex with dataclass modalities
```python
from docarray import Document, DocumentArray, dataclass
from docarray.typing import Image, Text


@dataclass
class MyDocument:
    image: Image
    paragraph: Text


_docs = [
    Document(MyDocument(image='image.png', paragraph='hello world')) for _ in range(10)
]
da = DocumentArray(
    _docs,
    config={'n_dim': 256},
    storage='annlite',
    subindex_configs={'@.[image]': {'n_dim': 512}, '@.[paragraph]': {'n_dim': 128}},
)
```
````
````{tab} Subindex with chunks
```python
from docarray import Document, DocumentArray

_docs = [
    Document(
        text='hello world',
        chunks=[Document(uri='image.png').load_uri_to_image_tensor()],
    )
    for _ in range(10)
]
da = DocumentArray(
    _docs,
    config={'n_dim': 256},
    storage='annlite',
    subindex_configs={'@c': {'n_dim': 512}},
)
```
````

The `subindex_configs` dictionary is structured in the following way:

- **Keys:** Each key in `subindex_configs` is the *name* of a subindex. It has to be a valid DocumentArray access path (such as `'@.[image]'`, `'@.[image, paragraph]'`, `'@c'`, or `'@cc'`).

- **Values:** Each value in `subindex_configs` is the *configuration* of a subindex. It can be any configuration that is valid for the given DocumentArray type.
Fields that are not given in the subindex configuration will be inherited from the parent configuration.


## Modify subindices

Once a DocumentArray with subindices has been constructed, any modifications to the parent DocumentArray will automatically update the subindices.

This means that you can insert, extend, delete etc. it like any other DocumentArray. For example:

````{tab} Subindex with dataclass modalities
```python
import numpy as np

# construct DocumentArry with subindices
da = DocumentArray(
    config={'n_dim': 256},
    storage='annlite',
    subindex_configs={'@.[image]': {'n_dim': 512}, '@.[paragraph]': {'n_dim': 128}},
)
# extend with Documents, including embeddings
_docs = [
    Document(MyDocument(image='image.png', paragraph='hello world')) for _ in range(10)
]
for d in _docs:
    d.image.embedding = np.random.rand(512)
    d.paragraph.embedding = np.random.rand(128)
with da:
    da.extend(_docs)
```
````
````{tab} Subindex with chunks
```python
import numpy as np

# construct DocumentArry with subindices
da = DocumentArray(
    config={'n_dim': 256},
    storage='annlite',
    subindex_configs={'@c': {'n_dim': 512}},
)
# extend with Documents, including embeddings
_docs = [
    Document(
        text='hello world',
        chunks=[Document(uri='image.png').load_uri_to_image_tensor()],
    )
    for _ in range(10)
]
for d in _docs:
    d.embedding = np.random.rand(256)
    d.chunks[0].embedding = np.random.rand(512)
with da:
    da.extend(_docs)
```
```` 

## Search through subindices

You can search through a subindex using the `on=` keyword in {meth}`~docarray.array.document.DocumentArray.find` and {meth}`~docarray.array.document.DocumentArray.match`:

````{tab} Subindex with dataclass modalities
```python
# find best matching images using .find()
top_image_matches = da.find(query=np.random.rand(512), on='@.[image]')
# find best matching paragraphs using .match()
Document(embedding=np.random.rand(128)).match(da, on='@.[paragraph]')
```
````
````{tab} Subindex with chunks
```python
# find best matching images using .find()
top_image_matches = da.find(query=np.random.rand(512), on='@c')
# find best matching images using .match()
Document(embedding=np.random.rand(512)).match(da, on='@c')
```
````

Such a search will return Documents from the subindex. If you are interested in the top-level Documents associated with
a match, you can retrieve them using `parent_id`:

````{tab} Subindex with dataclass modalities
```python
top_image_matches = da.find(query=np.random.rand(512), on='@.[image]')
top_level_matches = da[top_image_matches[:, 'parent_id']]
```
````
````{tab} Subindex with chunks
```python
top_image_matches = da.find(query=np.random.rand(512), on='@c')
top_level_matches = da[top_image_matches[:, 'parent_id']]
```
````