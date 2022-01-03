(recursive-nested-document)=
## Nested Structure

`Document` can be nested both horizontally and vertically. The following graphic illustrates the recursive `Document` structure. Each `Document` can have multiple "chunks"
and "matches", which are `Document` as well.

<img src="https://hanxiao.io/2020/08/28/What-s-New-in-Jina-v0-5/blog-post-v050-protobuf-documents.jpg">

|  Attribute   |   Description  |
| --- | --- |
| `doc.chunks` | The list of sub-Documents of this Document. They have `granularity + 1` but same `adjacency` |
| `doc.matches` | The list of matched Documents of this Document. They have `adjacency + 1` but same `granularity` |
| `doc.granularity` | The recursion "depth" of the recursive chunks structure |
| `doc.adjacency` | The recursion "width" of the recursive match structure |

You can add **chunks** (sub-Document) and **matches** (neighbour-Document) to a `Document`:

- Add in constructor:

  ```python
  d = Document(chunks=[Document(), Document()], matches=[Document(), Document()])
  ```

- Add to existing `Document`:

  ```python
  d = Document()
  d.chunks = [Document(), Document()]
  d.matches = [Document(), Document()]
  ```

- Add to existing `doc.chunks` or `doc.matches`:

  ```python
  d = Document()
  d.chunks.append(Document())
  d.matches.append(Document())
  ```

````{admonition} Note
:class: note
Both `doc.chunks` and `doc.matches` return `ChunkArray` and `MatchArray`, which are sub-classes
of {ref}`DocumentArray<documentarray>`. We will introduce `DocumentArray` later.
````

`````{admonition} Caveat: order matters
:class: alert


When adding sub-Documents to `Document.chunks`, avoid creating them in one line, otherwise the recursive Document structure will not be correct. This is because `chunks` use `ref_doc` to control their `granularity`. At `chunk` creation time the `chunk` doesn't know anything about its parent, and will get a wrong `granularity` value.

````{tab} ✅ Do
```python
from jina import Document

root_document = Document(text='i am root')
# add one chunk to root
root_document.chunks.append(Document(text='i am chunk 1'))
root_document.chunks.extend([
   Document(text='i am chunk 2'),
   Document(text='i am chunk 3'),
])  # add multiple chunks to root
```
````

````{tab} 😔 Don't
```python
from jina import Document

root_document = Document(
   text='i am root',
   chunks=[
      Document(text='i am chunk 2'),
      Document(text='i am chunk 3'),
   ]
)
```
````


`````
