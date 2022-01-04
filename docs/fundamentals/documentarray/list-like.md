# Other List-like Features

One can see `DocumentArray` as a Python list. Hence, many Python high-level iterator functions/tools can be used on `DocumentArray` as well.

## Shuffle

`DocumentArray` provides a `.shuffle` function that shuffles the entire `DocumentArray`. It accepts the parameter `seed`
.  `seed` helps you generate pseudo-random results. By default, `seed` is None.

To make use of the function:

```{code-block} python
---
emphasize-lines: 6, 7
---
from jina import Document, DocumentArray

da = DocumentArray()  # initialize a random DocumentArray
for idx in range(100):
    da.append(Document(id=idx))  # append 100 Documents into `da`
shuffled_da = da.shuffle()  # shuffle the DocumentArray
shuffled_da_with_seed = da.shuffle(seed=1)  # shuffle the DocumentArray with seed.
```

## Split by `.tags`

`DocumentArray` provides a `.split` function that splits the `DocumentArray` into multiple `DocumentArray`s according to the tag value (stored in `tags`) of each `Document`.
It returns a Python `dict` where `Documents` with the same `tag` value are grouped together, with their orders preserved from the original `DocumentArray`.

To make use of the function:

```{code-block} python
---
emphasize-lines: 10
---
from jina import Document, DocumentArray

da = DocumentArray()
da.append(Document(tags={'category': 'c'}))
da.append(Document(tags={'category': 'c'}))
da.append(Document(tags={'category': 'b'}))
da.append(Document(tags={'category': 'a'}))
da.append(Document(tags={'category': 'a'}))

rv = da.split(tag='category')
assert len(rv['c']) == 2  # category `c` is a DocumentArray has 2 Documents
```


## Iterate via `itertools`

As `DocumentArray` is an `Iterable`, you can also
use [Python's built-in `itertools` module](https://docs.python.org/3/library/itertools.html) on it. This enables
advanced "iterator algebra" on the `DocumentArray`.

For instance, you can group a `DocumentArray` by `parent_id`:

```{code-block} python
---
emphasize-lines: 5
---
from jina import DocumentArray, Document
from itertools import groupby

da = DocumentArray([Document(parent_id=f'{i % 2}') for i in range(6)])
groups = groupby(sorted(da, key=lambda d: d.parent_id), lambda d: d.parent_id)
for key, group in groups:
    key, len(list(group))
```

```text
('0', 3)
('1', 3)
```

## Filter

You can use Python's [built-in `filter()`](https://docs.python.org/3/library/functions.html#filter) to filter elements
in a `DocumentArray` object:

```{code-block} python
---
emphasize-lines: 8
---
from jina import DocumentArray, Document

da = DocumentArray([Document() for _ in range(6)])

for j in range(6):
    da[j].scores['metric'] = j

for d in filter(lambda d: d.scores['metric'].value > 2, da):
    print(d)
```

```text
{'id': 'b5fa4871-cdf1-11eb-be5d-e86a64801cb1', 'scores': {'values': {'metric': {'value': 3.0}}}}
{'id': 'b5fa4872-cdf1-11eb-be5d-e86a64801cb1', 'scores': {'values': {'metric': {'value': 4.0}}}}
{'id': 'b5fa4873-cdf1-11eb-be5d-e86a64801cb1', 'scores': {'values': {'metric': {'value': 5.0}}}}
```

You can build a `DocumentArray` object from the filtered results:

```python
from jina import DocumentArray, Document

da = DocumentArray([Document(weight=j) for j in range(6)])
da2 = DocumentArray(d for d in da if d.weight > 2)

print(da2)
```

```text
DocumentArray has 3 items:
{'id': '3bd0d298-b6da-11eb-b431-1e008a366d49', 'weight': 3.0},
{'id': '3bd0d324-b6da-11eb-b431-1e008a366d49', 'weight': 4.0},
{'id': '3bd0d392-b6da-11eb-b431-1e008a366d49', 'weight': 5.0}
```


## Sort

`DocumentArray` is a subclass of `MutableSequence`, therefore you can use Python's built-in `sort` to sort elements in
a `DocumentArray` object:

```{code-block} python
---
emphasize-lines: 11
---
from jina import DocumentArray, Document

da = DocumentArray(
    [
        Document(tags={'id': 1}),
        Document(tags={'id': 2}),
        Document(tags={'id': 3})
    ]
)

da.sort(key=lambda d: d.tags['id'], reverse=True)
print(da)
```

To sort elements in `da` in-place, using `tags[id]` value in a descending manner:

```text
<jina.types.arrays.document.DocumentArray length=3 at 5701440528>

{'id': '6a79982a-b6b0-11eb-8a66-1e008a366d49', 'tags': {'id': 3.0}},
{'id': '6a799744-b6b0-11eb-8a66-1e008a366d49', 'tags': {'id': 2.0}},
{'id': '6a799190-b6b0-11eb-8a66-1e008a366d49', 'tags': {'id': 1.0}}
```
