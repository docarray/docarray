(access-elements)=
# Access Documents

This is probably my favorite chapter so far. If you've come this far, you may be thinking: Okay, so you've re-implemented the Python List and called it DocumentArray. What's the big deal?

If it really were just a `list` and you can only access elements via `[1]`, `[-1]`, `[1:3]`, then you'd be right. However, DocumentArray offers _much_ more than simple indexing. It lets you fully exploit the rich and nested data structure of Documents in an easy and efficient way. 

The table below summarizes all the indexing routines that DocumentArray supports. You can use them to **get, set, and delete** items in a DocumentArray.

| Indexing routine                        | Example                                                                      | Return        |
|-----------------------------------------|------------------------------------------------------------------------------|---------------|
| by integer                              | `da[1]`, `da[-1]`                                                            | Document      |
| by integers                             | `da[1,2,3]`                                                                  | DocumentArray |
| by slice                                | `da[1:10:2]`, `da[5:]`                                                       | DocumentArray |
| by `id`                                 | `da['a04633546e6211ec8ad31e008a366d49']`                                     | Document      |
| by `id`s                                | `da['a04633546e6211ec8ad31e008a366d49', 'af7923406e6211ecbc811e008a366d49']` | DocumentArray |
| by boolean mask                         | `da[True, False, True, False] `                                              | DocumentArray |
| by Ellipsis                             | `da[...]`                                                                    | DocumentArray |
| by nested structure                     | `da['@cm,m,c']`, `da['@c1:3m']`, `da['@r[1]m[2]']`                           | DocumentArray |
| [by multimodal field](../../dataclass/) | `da['@.[banner]']`, `da['@.[banner].[image, summary]']`                      | DocumentArray |

Sounds exciting? Let's continue then.

````{tip}
Most of the examples below only show getting Documents for the sake of clarity. Note that you can always use the same syntax to get/set/delete Documents. For example:

```python
da = DocumentArray(...)

da[index]
da[index] = Document(...)
da[index] = DocumentArray(...)
del da[index]
```

````

## Basic indexing

Basic indexing such as by integer offset or slices are so common that we think they can go without saying. You can just use them like you would in a Python List:

```python
from docarray import DocumentArray

da = DocumentArray.empty(100)

da[0]
da[-1]
da[1:5]
da[1:100:10]
```

```text
<Document ('id',) at 834f14666e6511ec8e331e008a366d49>
<Document ('id',) at 834f32846e6511ec8e331e008a366d49>
<DocumentArray (length=4) at 4883468432>
<DocumentArray (length=10) at 4883468432>
```

## Index by Document `id`

A more interesting use case is selecting Documents by their `id`s:

```python
from docarray import DocumentArray

da = DocumentArray.empty(100)

print(da[0].id, da[1].id)
```

```text
7e27fa246e6611ec9a441e008a366d49 
7e27fb826e6611ec9a441e008a366d49
```

```python
print(da['7e27fa246e6611ec9a441e008a366d49'])
print(da['7e27fa246e6611ec9a441e008a366d49', '7e27fb826e6611ec9a441e008a366d49'])
```

```text
<Document ('id',) at 99851c7a6e6611ecba351e008a366d49>
<DocumentArray (length=2) at 4874066256>
```

No need to worry about efficiency here: It's `O(1)`.

Based on the same technique, you can check if a Document is inside a DocumentArray using Python's `in` syntax:

```python
from docarray import DocumentArray, Document

da = DocumentArray.empty(10)

da[0] in da
Document() in da
```

```text
True
False
```


## Index by boolean mask

Using a boolean mask to select Documents is useful for updating or filtering certain Documents:

```python
from docarray import DocumentArray

da = DocumentArray.empty(100)
mask = [True, False] * 50

del da[mask]

print(da)
```

```text
<DocumentArray (length=50) at 4513619088>
```

Note that if the boolean mask's length is smaller than the DocumentArray's length, the remaining part is padded to `False`. 

(path-string)=
## Index by nested structure

From an earlier chapter, we already know {ref}`Documents can be nested<recursive-nested-document>`. DocumentArray provides makes it easy to traverse over the nested structure and select Documents:

```python
da['@path1,path2,path3']
```

- The path-string must start with `@`.
- Multiple paths are separated by commas `,`.
- A path represents the route from the top-level Documents to the destination. Use `c` to select chunks, `cc` to select chunks of chunks, `m` to select matches, `mc` to select chunks of matches, `r` to select top-level Documents.
- A path can only go deeper, not shallower. You can use commas `,` to start a new path from the very top-level.
- Optionally, specifying a slice or offset at each level (for example, `r[-1]m[:3]`) selects the first 3 matches of the last root document.

```{seealso}
If you're working with a DocumentArray that was created through DocArray's {ref}`dataclass <dataclass>` API,
you can also directly access sub-documents by specifying the modality name that you assigend to them.

To see how to do that, see {ref}`here <mm-access-da>`.
```

Let's practice. First construct a DocumentArray with nested Documents:

```python
from docarray import DocumentArray

da = DocumentArray().empty(3)
for d in da:
    d.chunks = DocumentArray.empty(2)
    d.matches = DocumentArray.empty(2)

da.summary()
```

```text
                    Documents Summary                    
                                                         
  Length                    3                            
  Homogenous Documents      True                         
  Has nested Documents in   ('chunks', 'matches')        
  Common Attributes         ('id', 'chunks', 'matches')  
                                                         
                        Attributes Summary                        
                                                                  
  Attribute   Data type         #Unique values   Has empty value  
 ──────────────────────────────────────────────────────────────── 
  chunks      ('ChunkArray',)   3                False            
  id          ('str',)          3                False            
  matches     ('MatchArray',)   3                False  
```

This simple DocumentArray contains three Documents, each of which contains two matches and two chunks. Let's plot one of them.

```text
 <Document ('id', 'chunks', 'matches') at 2f94c1426ee511ecbb491e008a366d49>
    └─ matches
          ├─ <Document ('id', 'adjacency') at 2f94cd9a6ee511ecbb491e008a366d49>
          └─ <Document ('id', 'adjacency') at 2f94cdfe6ee511ecbb491e008a366d49>
    └─ chunks
          ├─ <Document ('id', 'parent_id', 'granularity') at 2f94c4086ee511ecbb491e008a366d49>
          └─ <Document ('id', 'parent_id', 'granularity') at 2f94c46c6ee511ecbb491e008a366d49>
```

That's still too much information, let's minimize it:

```{figure} images/docarray-index-example.svg
:width: 10%
```

Now let's use the red dot to depict our intended selection. Here's where we use the path-syntax:

```{figure} images/docarray-index-example-full1.svg
```

```python
print(da['@m'])
print(da['@c'])
print(da['@c,m'])
print(da['@c,m,r'])
```

```text
<DocumentArray (length=6) at 4912623312>
<DocumentArray (length=6) at 4905929552>
<DocumentArray (length=12) at 4913359824>
<DocumentArray (length=15) at 4912623312>
```

Let's now consider a deeper nested structure and use the path syntax to select Documents:

```{figure} images/docarray-index-example-full2.svg
```

Last but not the least, you can use integer, or integer slice to restrict the selection:
```{figure} images/docarray-index-example-full3.svg
:width: 60%
```

This is useful to get the top matches of all matches from all Documents:

```python
da['@m[:5]']
```

You can add spaces in the path-string for better readability.

## Index by flatten

What if I just want a flat DocumentArray without all nested structure? Can I select all Documents regardless of their nested structure?

Yes! Simply use the ellipsis literal as the selector `da[...]`:

```python
from docarray import DocumentArray

da = DocumentArray().empty(3)
for d in da:
    d.chunks = DocumentArray.empty(2)
    d.matches = DocumentArray.empty(2)

da[...].summary()
```

```text
                           Documents Summary                           
                                                                       
  Length                           15                                  
  Homogenous Documents             False                               
  6 Documents have attributes      ('id', 'parent_id', 'granularity')  
  6 Documents have attributes      ('id', 'adjacency')                 
  3 Documents have one attribute   ('id',)                             
                                                                       
                      Attributes Summary                      
                                                              
  Attribute     Data type   #Unique values   Has empty value  
 ──────────────────────────────────────────────────────────── 
  adjacency     ('int',)    2                False            
  granularity   ('int',)    2                False            
  id            ('str',)    15               False            
  parent_id     ('str',)    4                False 
```

Note that there are no `chunks` or `matches` in any of the Documents from `da[...]` anymore. They have all been flattened.

Documents in `da[...]` are in the chunks-and-depth-first order, i.e depth-first traversing to all chunks and then to all matches.


## Other handy helpers

### Batching


```{tip}
To batch and process a DocumentArray in parallel in a non-blocking way, use {meth}`~docarray.array.mixins.parallel.ParallelMixin.map_batch` and refer to {ref}`map-batch`.
```

You can batch a large DocumentArray into smaller ones with {meth}`~docarray.array.mixins.group.GroupMixin.batch`. This is useful when a DocumentArray is too big to process at once.

```python
from docarray import DocumentArray

da = DocumentArray.empty(1000)

for b_da in da.batch(batch_size=256):
    print(b_da)
```

```text
<DocumentArray (length=256) at 4887691536>
<DocumentArray (length=256) at 4887691600>
<DocumentArray (length=256) at 4887691408>
<DocumentArray (length=232) at 4887691536>
```

### Sampling

```python
from docarray import DocumentArray

da = DocumentArray.empty(1000).sample(10)
``` 

```text
<DocumentArray (length=10) at 4887691536>
```


### Shuffling

To shuffle a DocumentArray in-place:

```python
from docarray import DocumentArray

da = DocumentArray.empty(1000)
da.shuffle()
```

### Splitting by `.tags`

You can split a DocumentArray into multiple DocumentArrays according to a tag value (stored in `tags`) of each Document.
It returns a Python `dict` where Documents with the same `tag` value are grouped together in a new DocumentArray, with their orders preserved from the original DocumentArray.

```python
from docarray import Document, DocumentArray

da = DocumentArray(
    [
        Document(tags={'category': 'c'}),
        Document(tags={'category': 'c'}),
        Document(tags={'category': 'b'}),
        Document(tags={'category': 'a'}),
        Document(tags={'category': 'a'}),
    ]
)

rv = da.split_by_tag(tag='category')
```

```text
{'c': <DocumentArray (length=2) at 4869273936>, 
 'b': <DocumentArray (length=1) at 4876081680>, 
 'a': <DocumentArray (length=2) at 4876735056>}
```

## What's next?

Now you know how to select Documents from DocumentArray, next you'll learn how to {ref}`select attributes from DocumentArray<bulk-access>`. Spoiler alert: it follows the same syntax. 
