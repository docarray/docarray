# Access Elements

Like a `List` *and* a `Dict`, elements in `DocumentArray` can be accessed via integer index, string `id` or `slice` indices:

```python
from jina import DocumentArray, Document

da = DocumentArray([Document(id='hello'), Document(id='world'), Document(id='goodbye')])

da[0]
da[1:2]
da['world']
```

```text
<jina.types.document.Document id=hello at 5699749904>
<jina.types.arrays.document.DocumentArray length=1 at 5705863632>
<jina.types.document.Document id=world at 5736614992>
```

```{tip}
To access Documents with nested Documents, please refer to {ref}`traverse-doc`.
```


## Traverse nested elements

{meth}`~jina.types.arrays.mixins.traverse.TraverseMixin.traverse_flat` function is an extremely powerful tool for iterating over nested and recursive Documents. You get a generator as the return value, which generates `Document`s on the provided traversal paths. You can use or modify `Document`s and the change will be applied in-place. 


### Syntax of traversal path

`.traverse_flat()` function accepts a `traversal_paths` string which can be defined as follow:

```text
path1,path2,path3,...
```

```{tip}
Its syntax is similar to `subscripts` in [`numpy.einsum()`](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html), but without `->` operator.
```

Note that,
- paths are separated by comma `,`;
- each path is a string represents a path from the top-level `Document`s to the destination. You can use `c` to select chunks, `m` to select matches;
- a path can be a single letter, e.g. `c`, `m` or multi-letters, e.g. `ccc`, `cmc`, depending on how deep you want to go;
- to select top-level `Document`s, you can use `r`;
- a path can only go deep, not go back. You can use comma `,` to "reset" the path back to the very top-level;

### Example

Let's look at an example. Assume you have the following `Document` structure:

````{dropdown} Click to see the construction of the nested Document
```python
from jina import DocumentArray, Document

root = Document(id='r1')

chunk1 = Document(id='r1c1')
root.chunks.append(chunk1)
root.chunks[0].matches.append(Document(id='r1c1m1'))

chunk2 = Document(id='r1c2')
root.chunks.append(chunk2)
chunk2_chunk1 = Document(id='r1c2c1')
chunk2_chunk2 = Document(id='r1c2c2')
root.chunks[1].chunks.extend([chunk2_chunk1, chunk2_chunk2])
root.chunks[1].chunks[0].matches.extend([Document(id='r1c2c1m1'), Document(id='r1c2c1m2')])

chunk3 = Document(id='r1c3')
root.chunks.append(chunk3)

da = DocumentArray([root])
root.plot()
```
````

```{figure} traverse-example-docs.svg
:align: center
```

Now one can use `da.traverse_flat('c')` To get all the `Chunks` of the root `Document`; `da.traverse_flat('m')` to can get all the `Matches` of the root `Document`.

This allows us to composite the `c` and `m` to find `Chunks`/`Matches` which are in a deeper level:

- `da.traverse_flat('cm')` will find all `Matches` of the `Chunks` of root `Document`.
- `da.traverse_flat('cmc')` will find all `Chunks` of the `Matches` of `Chunks` of root `Document`.
- `da.traverse_flat('c,m')` will find all `Chunks` and `Matches` of root `Document`.

````{dropdown} Examples

```python
for ma in da.traverse_flat('cm'):
  for m in ma:
    print(m.json())
```

```json
{
  "adjacency": 1,
  "granularity": 1,
  "id": "r1c1m1"
}
```

```python
for ma in da.traverse_flat('ccm'):
  for m in ma:
    print(m.json())
```

```json
{
  "adjacency": 1,
  "granularity": 2,
  "id": "r1c2c1m1"
}
{
  "adjacency": 1,
  "granularity": 2,
  "id": "r1c2c1m2"
}
```

```python
for ma in da.traverse('cm', 'ccm'):
  for m in ma:
    print(m.json())
```

```json
{
  "adjacency": 1,
  "granularity": 1,
  "id": "r1c1m1"
}
{
  "adjacency": 1,
  "granularity": 2,
  "id": "r1c2c1m1"
}
{
  "adjacency": 1,
  "granularity": 2,
  "id": "r1c2c1m2"
}
```
````

When calling `da.traverse_flat('cm,ccm')` the result in our example will be:

```text
DocumentArray([
    Document(id='r1c1m1', adjacency=1, granularity=1),
    Document(id='r1c2c1m1', adjacency=1, granularity=2),
    Document(id='r1c2c1m2', adjacency=1, granularity=2)
])
```

{meth}`jina.types.arrays.mixins.traverse.TraverseMixin.traverse_flat_per_path` is another method for `Document` traversal. It works
like `traverse_flat` but groups `Documents` into `DocumentArrays` based on traversal path. When
calling `da.traverse_flat_per_path('cm,ccm')`, the resulting generator yields the following `DocumentArray`:

```text
DocumentArray([
    Document(id='r1c1m1', adjacency=1, granularity=1),
])

DocumentArray([
    Document(id='r1c2c1m1', adjacency=1, granularity=2),
    Document(id='r1c2c1m2', adjacency=1, granularity=2)
])
```

### Flatten Document

If you simply want to traverse **all** chunks and matches regardless their levels. You can simply use {meth}`~jina.types.arrays.mixins.traverse.TraverseMixin.flatten`. It will return a `DocumentArray` with all chunks and matches flattened into the top-level, no more nested structure.


## Batching

One can batch a large `DocumentArray` into small ones via {func}`~jina.types.arrays.mixins.group.GroupMixin.batch`. This is useful when a `DocumentArray` is too big to process at once. It is particular useful on `DocumentArrayMemmap`, which ensures the data gets loaded on-demand and in a conservative manner.

```python
from jina import DocumentArray

da = DocumentArray.empty(1000)

for b_da in da.batch(batch_size=256):
    print(len(b_da))
```

```text
256
256
256
232
```

```{tip}
For processing batches in parallel, please refer to {meth}`~jina.types.arrays.mixins.parallel.ParallelMixin.map_batch`.
```

## Sampling

`DocumentArray` provides a `.sample` function that samples `k` elements without replacement. It accepts two parameters, `k`
and `seed`. `k` defines the number of elements to sample, and `seed`
helps you generate pseudo-random results. Note that `k` should always be less than or equal to the length of the
`DocumentArray`.

To make use of the function:

```{code-block} python
---
emphasize-lines: 6, 7
---
from jina import Document, DocumentArray

da = DocumentArray()  # initialize a random DocumentArray
for idx in range(100):
    da.append(Document(id=idx))  # append 100 Documents into `da`
sampled_da = da.sample(k=10)  # sample 10 Documents
sampled_da_with_seed = da.sample(k=10, seed=1)  # sample 10 Documents with seed.
```