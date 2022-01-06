(access-elements)=
# Access Elements

This is probably my favorite chapter so far. Readers come to this far may ask: okay you re-implement Python List via DocumentArray, what's the big deal?

If it is just a `list` and you can only access elements via `[1]`, `[-1]`, `[1:3]`, then it is no big deal. However, DocumentArray offers much more than simple indexing. It allows you to fully exploit the rich & nested data structure of Document in an easy and efficient way. 

The table below summarizes all indexing routines that DocumentArray supports. You can use them to **get, set, and delete** items in DocumentArray.

| Indexing routine    | Example                                                                      | Return         |
|---------------------|------------------------------------------------------------------------------|----------------|
| by integer          | `da[1]`, `da[-1]`                                                            | Document       |
| by integers         | `da[1,2,3]`                                                                  | DocumentArray  |
| by slice            | `da[1:10:2]`, `da[5:]`                                                       | DocumentArray  |
| by `id`             | `da['a04633546e6211ec8ad31e008a366d49']`                                     | Document       |
| by `id`s            | `da['a04633546e6211ec8ad31e008a366d49', 'af7923406e6211ecbc811e008a366d49']` | DocumentArray  |
| by boolean mask     | `da[True, False, True, False] `                                              | DocumentArray  |
| by Ellipsis         | `da[...]`                                                                    | DocumentArray |
| by nested structure | `da['@cm,m,c']`, `da['@c1:3m']`                                              | DocumentArray |

Sounds exciting? Let's continue then.

````{tip}
Most of the examples below only show getting Documents for the sake of clarity. Note that you can always use the same syntax for get/set/delete Documents. For example,

```python
da = DocumentArray(...)

da[index]
da[index] = DocumentArray(...)
del da[index]
```

````

## Basic indexing

Basic indexing such as by the integer offset, the slices are so common that I don't think we need more words. You can just use it as in Python List.

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

A more interesting one is selecting Documents by their `id`. 

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

No need to worry about efficiency here, it is `O(1)`.

## Index by boolean mask

You can use a boolean mask to select Documents. This becomes useful when you want to update or filter our certain Documents:

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

## Index by nested structure

From early chapter, we already know {ref}`Document can be nested<recursive-nested-document>`. DocumentArray provides very easy way to traverse over the nested structure and select Documents. All you need to do is following the syntax below:

```python
da['@path1,path2,path3']
```

Note that,
- the path string must starts with `@`
- multiple paths are separated by comma `,`;
- a path is a string represents the route from the top-level Documents to the destination. You can use `c` to select chunks, `cc` to select chunks of the chunks, `m` to select matches, `mc` to select matches of the chunks, `r` to select the top-level Documents.
- a path can only go deep, not go back. You can use comma `,` to start a new path from the very top-level.


## Index by flatten

What if I just want a flat DocumentArray without all nested structure, can I select all Documents regardless their nested structure?

Yes! Simply use ellipsis literal as the selector `da[...]`:

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

Note that there is no `chunks` and `matches` in any of the Document from `da[...]` anymore. They are all flattened.

## Batching

One can batch a large DocumentArray into small ones via {meth}`~docarray.array.mixins.group.GroupMixin.batch`. This is useful when a DocumentArray is too big to process at once.

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

## Sampling

```python
from docarray import DocumentArray

da = DocumentArray.empty(1000).sample(10)
``` 

```text
<DocumentArray (length=10) at 4887691536>
```