(subindex)=
# Search over Nested Structure

To use {meth}`~docarray.array.mixins.find.FindMixin.find` on multimodal or nested Documents (a multimodal Document is intrinsically a nested Document), you need "subindices". The word "subindices" represents that you are adding a new sub-level of indexing to the DocumentArray and making it searchable.

Each subindex indexes and stores one nesting level (like `'@c'` or a {ref}`custom modality <dataclass>` like `'@.[image]'`) and makes it directly searchable. Under the hood, subindices are fully fledged DocumentArrays with their own {ref}`document store<doc-store>`.

```{seealso}
To see subindices in action, check {ref}`here <multimodal-search-example>`.
```

## Construct subindices

You can specify subindices when you create a DocumentArray
by passing configuration for each desired subindex to the `subindex_configs` parameter:

````{tab} Subindex with dataclass modalities
```python
from docarray import Document, DocumentArray, dataclass
from docarray.typing import Image, Text


@dataclass
class MyDocument:
    image: Image
    paragraph: Text


_docs = [
    Document(
        MyDocument(
            image='https://docarray.jina.ai/_images/apple.png', paragraph='hello world'
        )
    )
    for _ in range(10)
]
da = DocumentArray(
    _docs,
    config={'n_dim': 256},
    storage='annlite',
    subindex_configs={'@.[image]': {'n_dim': 512}, '@.[paragraph]': {'n_dim': 128}},
)
```

```console
╭───────────────────── Documents Summary ─────────────────────╮
│                                                             │
│   Length                    10                              │
│   Homogenous Documents      True                            │
│   Has nested Documents in   ('chunks',)                     │
│   Common Attributes         ('id', 'embedding', 'chunks')   │
│   Multimodal dataclass      True                            │
│                                                             │
╰─────────────────────────────────────────────────────────────╯
╭──────────────────────── Attributes Summary ────────────────────────╮
│                                                                    │
│   Attribute   Data type         #Unique values   Has empty value   │
│  ────────────────────────────────────────────────────────────────  │
│   chunks      ('ChunkArray',)   10               False             │
│   embedding   ('ndarray',)      10               False             │
│   id          ('str',)          10               False             │
│                                                                    │
╰────────────────────────────────────────────────────────────────────╯
╭────── DocumentArrayAnnlite Config ──────╮
│                                         │
│   n_dim              256                │
│   metric             cosine             │
│   serialize_config   {}                 │
│   data_path          /tmp/tmp_w1yqmpc   │
│   ef_construction    None               │
│   ef_search          None               │
│   max_connection     None               │
│   columns            {}                 │
│                                         │
╰─────────────────────────────────────────╯
```

````
````{tab} Subindex with chunks
```python
from docarray import Document, DocumentArray

_docs = [
    Document(
        text='hello world',
        chunks=[
            Document(
                uri='https://docarray.jina.ai/_images/apple.png'
            ).load_uri_to_image_tensor()
        ],
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

```console
╭───────────────────────── Documents Summary ─────────────────────────╮
│                                                                     │
│   Length                    10                                      │
│   Homogenous Documents      True                                    │
│   Has nested Documents in   ('chunks',)                             │
│   Common Attributes         ('id', 'text', 'embedding', 'chunks')   │
│   Multimodal dataclass      False                                   │
│                                                                     │
╰─────────────────────────────────────────────────────────────────────╯
╭──────────────────────── Attributes Summary ────────────────────────╮
│                                                                    │
│   Attribute   Data type         #Unique values   Has empty value   │
│  ────────────────────────────────────────────────────────────────  │
│   chunks      ('ChunkArray',)   10               False             │
│   embedding   ('ndarray',)      10               False             │
│   id          ('str',)          10               False             │
│   text        ('str',)          1                False             │
│                                                                    │
╰────────────────────────────────────────────────────────────────────╯
╭────── DocumentArrayAnnlite Config ──────╮
│                                         │
│   n_dim              256                │
│   metric             cosine             │
│   serialize_config   {}                 │
│   data_path          /tmp/tmp_iar4ofr   │
│   ef_construction    None               │
│   ef_search          None               │
│   max_connection     None               │
│   columns            {}                 │
│                                         │
╰─────────────────────────────────────────╯
```
````

The `subindex_configs` dictionary is structured as follows:

- **Keys:** Each key in `subindex_configs` is the *name* of a subindex. It must be a valid DocumentArray access path (like `'@.[image]'`, `'@.[image, paragraph]'`, `'@c'`, or `'@cc'`).

- **Values:** Each value in `subindex_configs` is the *configuration* of a subindex. It can be any valid configuration for the given DocumentArray type.
Fields that are not given in the subindex configuration are inherited from the parent configuration.

## Modify subindices

Once you've constructed a DocumentArray with subindices, modifying the parent DocumentArray automatically updates the subindices.

This means you can insert, extend, delete (etc.) it like any other DocumentArray:

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

Such a search will return Documents from the subindex. If you are interested in the top-level Documents associated with a match, you can retrieve them by setting `return_root=True` in `find`:

````{tab} Subindex with dataclass modalities
```python
top_level_matches = da.find(query=np.random.rand(512), on='@.[image]', return_root=True)
```
````
````{tab} Subindex with chunks
```python
top_level_matches = da.find(query=np.random.rand(512), on='@c', return_root=True)
```
````

````{admonition} Note
:class: note
When you add or change Documents directly on a subindex, the `_root_id_` (or `parent_id` for DocumentArrayInMemory) of new Documents should be set manually for `return_root=True` to work:

```python
da['@c'].extend(
    Document(embedding=np.random.random(512), tags={'_root_id_': 'your_root_id'})
)
```
````
