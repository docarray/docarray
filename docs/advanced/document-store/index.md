# Document Store

```{toctree}
:hidden:

weaviate
sqlite
qdrant
annlite

```

Documents inside a DocumentArray can live in a [document store](https://en.wikipedia.org/wiki/Document-oriented_database) instead of in memory, e.g. in SQLite, Redis. Comparing to the in-memory storage, the benefit of using an external store is often about longer persistence and faster retrieval. 

The look-and-feel of a DocumentArray with external store is **almost the same** as a regular in-memory DocumentArray. This allows users to easily switch between backends under the same DocArray idiom.  

Take SQLite as an example, using it as the store backend of a DocumentArray is as simple as follows:

```python
from docarray import DocumentArray, Document

da = DocumentArray(storage='sqlite', config={'connection': 'example.db'})

da.append(Document())
da.summary()
```

```text
        Documents Summary         
                                  
  Length                 1        
  Homogenous Documents   True     
  Common Attributes      ('id',)  
                                  
                     Attributes Summary                     
                                                            
  Attribute   Data type   #Unique values   Has empty value  
 ────────────────────────────────────────────────────────── 
  id          ('str',)    1                False            
                                                           
```

Creating, retrieving, updating, deleting Documents are identical to the regular {ref}`DocumentArray<documentarray>`. All DocumentArray methods such as `.summary()`, `.embed()`, `.plot_embeddings()` should work out of the box.

## Construct

There are two ways for initializing a DocumentArray with a store backend.

````{tab} Specify storage

```python
from docarray import DocumentArray

da = DocumentArray(storage='sqlite')
```


```text
<DocumentArray[SQLite] (length=0) at 4477814032>
```
````

````{tab} Import the class and alias it  

```python
from docarray.array.sqlite import DocumentArraySqlite as DocumentArray

da = DocumentArray()
```

```text
<DocumentArray[SQLite] (length=0) at 4477814032>
```

````

Depending on the context, you can choose the style that fits better. For example, if one wants to use class method such as `DocumentArray.empty(10)`, then explicit importing `DocumentArraySqlite` is the way to go. Of course, you can choose not to alias the imported class to make the code even more explicit.

### Construct with config

The config of a store backend is either store-specific dataclass object or a `dict` that can be parsed into the former.

One can pass the config in the constructor via `config`:

````{tab} Use dataclass

```python
from docarray import DocumentArray
from docarray.array.sqlite import SqliteConfig

cfg = SqliteConfig(connection='example.db', table_name='test')

da = DocumentArray(storage='sqlite', config=cfg)
```

````

````{tab} Use dict

```python
from docarray import DocumentArray

da = DocumentArray(storage='sqlite', config={'connection': 'example.db', table_name='test'})
```

````

Using dataclass gives you better type-checking in IDE but requires an extra import; using dict is more flexible but can be error-prone. You can choose the style that fits best to your context.

## Known limitations

### Out-of-array modification

One can not take a Document *out* from a DocumentArray and modify it, then expect its modification to be committed back to the DocumentArray.

Specifically, the pattern below is not supported by any external store backend:

```python
from docarray import DocumentArray

da = DocumentArray(storage='any_store_beyond_in_memory')
d = da[0]  # or any access-element method, e.g. by id, by slice
d.text = 'hello'

print(da[0])  # this will NOT change to `hello`
```

The solution is simple: use {ref}`column-selector<bulk-access>`:

```python
da[0, 'text'] = 'hello'
```

### Elements access is slower

Obviously, a DocumentArray with on-disk storage is slower than in-memory DocumentArray. However, if you choose to use on-disk storage, then often your concern of persistence overwhelms the concern of efficiency.
 
Slowness can affect all functions of DocumentArray. On the bright side, they may not be that severe as you would expect. Modern database are highly optimized. Moreover, some database provides faster method for resolving certain queries, e.g. nearest-neighbour queries. We are actively and continuously improving DocArray to better leverage those features. 
