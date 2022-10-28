(recursive-nested-document)=
# Nested Structure

Document can be nested both horizontally and vertically via `.matches` and `.chunks`. The picture below illustrates the recursive Document structure. 

```{figure} images/nested-structure.svg
```

|  Attribute   | Description                                                                                     |
| --- |-------------------------------------------------------------------------------------------------|
| `doc.chunks` | The list of sub-Documents of this Document. They have `granularity + 1` but same `adjacency`    |
| `doc.matches` | The list of matched Documents of this Document. They have `adjacency + 1` but same `granularity` |
| `doc.granularity` | The "depth" of the nested chunks structure                                             |
| `doc.adjacency` | The "width" of the nested match structure                                             |

You can add **chunks** (sub-Document) and **matches** (neighbour-Document) to a Document:

- Add in constructor:

  ```python
  d = Document(chunks=[Document(), Document()], matches=[Document(), Document()])
  ```

- Add to existing Document:

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

Both `doc.chunks` and `doc.matches` return {ref}`DocumentArray<documentarray>`.

To get a clear picture of a nested Document, use {meth}`~docarray.document.mixins.plot.PlotMixin.summary`, e.g.:

```python
d.summary()
```

```text
 <Document ('id', 'chunks', 'matches') at 7f907d786d6c11ec840a1e008a366d49>
    └─ matches
          ├─ <Document ('id', 'adjacency') at 7f907c606d6c11ec840a1e008a366d49>
          └─ <Document ('id', 'adjacency') at 7f907cba6d6c11ec840a1e008a366d49>
    └─ chunks
          ├─ <Document ('id', 'parent_id', 'granularity') at 7f907ab26d6c11ec840a1e008a366d49>
          └─ <Document ('id', 'parent_id', 'granularity') at 7f907c106d6c11ec840a1e008a366d49>
```

## What's next?

When you have multiple Documents with nested structures, traversing over certain chunks and matches can be crucial. Fortunately, this is extremely simple thanks to DocumentArray as shown in {ref}`access-elements`.

Note that some methods rely on these two attributes, some methods require these two attributes to be filled in advance. For example, {meth}`~docarray.array.mixins.match.MatchMixin.match` will fill `.matches`, whereas {meth}`~docarray.array.mixins.evaluation.EvaluationMixin.evaluate` requires `.matches` to be filled.


