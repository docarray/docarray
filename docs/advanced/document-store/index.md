(doc-store)=
# Document Store

```{toctree}
:hidden:

weaviate
sqlite
qdrant
annlite
elasticsearch
subindex
extend
benchmark
```

Documents inside a DocumentArray can live in a [document store](https://en.wikipedia.org/wiki/Document-oriented_database) instead of in memory, e.g. in SQLite, Redis.
Comparing to the in-memory storage, the benefit of using an external store is often about longer persistence and faster retrieval. 

The look-and-feel of a DocumentArray with external store is **almost the same** as a regular in-memory DocumentArray. This allows users to easily switch between backends under the same DocArray idiom.  

Take SQLite as an example, using it as the store backend of a DocumentArray is as simple as follows:

```python
from docarray import DocumentArray, Document

da = DocumentArray(storage='sqlite', config={'connection': 'example.db'})

with da:
    da.append(Document())
da.summary()
```

```text
╭──────── Documents Summary ─────────╮
│                                    │
│   Length                 1         │
│   Homogenous Documents   True      │
│   Common Attributes      ('id',)   │
│   Multimodal dataclass   False     │
│                                    │
╰────────────────────────────────────╯
╭───────────────────── Attributes Summary ─────────────────────╮
│                                                              │
│   Attribute   Data type   #Unique values   Has empty value   │
│  ──────────────────────────────────────────────────────────  │
│   id          ('str',)    1                False             │
│                                                              │
╰──────────────────────────────────────────────────────────────╯
╭──────────────────────── DocumentArraySqlite Config ────────────────────────╮
│                                                                            │
│   connection         example.db                                            │
│   table_name         DocumentArraySqlite97c8c833586444a89272ff0ff4287edb   │
│   serialize_config   {}                                                    │
│   conn_config        {}                                                    │
│   journal_mode       DELETE                                                │
│   synchronous        OFF                                                   │
│                                                                            │
╰────────────────────────────────────────────────────────────────────────────╯
```
Note that  `da` was modified inside a `with` statement. This context manager ensures that the the `DocumentArray` indices,
which allow users to access the  `DocumentArray` by position (allowing statements such as `da[1]`),
are properly mapped and saved to the storage backend.
This is the recommended default usage to modify a DocumentArray that lives on a document store to avoid
unexpected behaviors that can yield to, for example, inaccessible elements by position.


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

```{admonition} Subindices
:class: seealso

When working with multimodal or nested data, you often want to perform search on a particular modality or nesting.

To enable this without having to load data into memory, you can construct your DocumentArray with a subindex.
To learn how to do that, see {ref}`here <subindex>`.

```

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

da = DocumentArray(
    storage='sqlite', config={'connection': 'example.db', 'table_name': 'test'}
)
```

````

Using dataclass gives you better type-checking in IDE but requires an extra import; using dict is more flexible but can be error-prone. You can choose the style that fits best to your context.


## Search summary

DocArray supports multiple storage backends with different search features. The following table showcases relevant functionalities that are supported (✅) or not supported (❌) in DocArray depending on the backend:


| Name                                                           | Construction                             |  vector search | vector search + filter | filter|
|----------------------------------------------------------------|------------------------------------------|---------------|------------------------|---------------|
| [`None`](./../../../fundamentals/documentarray/#documentarray) | `DocumentArray()`                        | ✅             | ✅                      | ✅             |
| [`Sqlite`](./sqlite/#sqlite)                                   | `DocumentArray(storage='sqlite')`        | ❌             | ❌                      | ✅             | 
| [`Weaviate`](./weaviate/#weaviate)                             | `DocumentArray(storage='weaviate')`      | ✅             | ✅                      | ✅             |
| [`Qdrant`](./qdrant/#qdrant)                                   | `DocumentArray(storage='qdrant')`        | ✅             | ✅                      | ❌             |
| [`Annlite`](./annlite/#annlite)                                | `DocumentArray(storage='annlite')`        | ✅             | ✅                      | ✅             |
| [`ElasticSearch`](./elasticsearch/#elasticsearch)              | `DocumentArray(storage='elasticsearch')` | ✅             | ✅                      | ✅             |


Here we understand by 

- **vector search**: perform approximate nearest neighbour search (or exact full scan search). The input of the search function is a numpy array or a DocumentArray containing an embedding. 

- **vector search + filter**: perform approximate nearest neighbour search (or exact full scan search). The input of the search function is a numpy array or a DocumentArray containing an embedding and a filter. 

- **filter**: perform a filter step over the data. The input of the search function is a filter. 

The capabilities of  **vector search**,  **vector search + filter** can be used using the  {meth}`~docarray.array.mixins.find.FindMixin.find` or {func}`~docarray.array.mixins.match.MatchMixin.match` methods thorugh a  `DocumentArray`.
The **filter** functionality is available using the `.find` method in a `DocumentArray`. 
A detailed explanation of the differences between `.find` and `.match` can be found [`here`](./../../../fundamentals/documentarray/matching) 


### Vector search example

Example of  **vector search**

````{tab} .find

```python
from docarray import Document, DocumentArray
import numpy as np

n_dim = 3
da = DocumentArray(
    storage='annlite',
    config={'n_dim': n_dim, 'metric': 'Euclidean'},
)

with da:
    da.extend([Document(embedding=i * np.ones(n_dim)) for i in range(10)])

result = da.find(np.array([2, 2, 2]), limit=6)
result[:, 'embedding']
```
````

````{tab} .match

```python
from docarray import Document, DocumentArray
import numpy as np

n_dim = 3
da = DocumentArray(
    storage='annlite',
    config={'n_dim': n_dim, 'metric': 'Euclidean'},
)

with da:
    da.extend([Document(embedding=i * np.ones(n_dim)) for i in range(10)])

query = Document(embedding=np.array([2, 2, 2]))
query.match(da, limit=6)
query.matches[:, 'embedding']
```
````

```text
array([[2., 2., 2.],
       [1., 1., 1.],
       [3., 3., 3.],
       [0., 0., 0.],
       [4., 4., 4.],
       [5., 5., 5.]])
```

### Vector search with filter example

Example of **vector search + filter**

````{tab} .find

```python
from docarray import Document, DocumentArray
import numpy as np

n_dim = 3
metric = 'Euclidean'

da = DocumentArray(
    storage='annlite',
    config={'n_dim': n_dim, 'columns': [('price', 'float')], 'metric': metric},
)

with da:
    da.extend(
        [
            Document(id=f'r{i}', embedding=i * np.ones(n_dim), tags={'price': i})
            for i in range(10)
        ]
    )

max_price = 3
n_limit = 4

filter = {'price': {'$lte': max_price}}
query = np.array([2, 2, 2])
results = da.find(query=query, filter=filter)
results[:, 'embedding']
```
````

````{tab} .match

```python
from docarray import Document, DocumentArray
import numpy as np

n_dim = 3
metric = 'Euclidean'

da = DocumentArray(
    storage='annlite',
    config={'n_dim': n_dim, 'columns': [('price', 'float')], 'metric': metric},
)

with da:
    da.extend(
        [
            Document(id=f'r{i}', embedding=i * np.ones(n_dim), tags={'price': i})
            for i in range(10)
        ]
    )

max_price = 3
n_limit = 4

filter = {'price': {'$lte': max_price}}
query = Document(embedding=np.array([2, 2, 2]))
query.match(da, filter=filter)
query.matches[:, 'embedding']
```
````

```text
array([[2., 2., 2.],
       [1., 1., 1.],
       [3., 3., 3.],
       [0., 0., 0.]])
```

### Filter example

Example of **filter**

```python
from docarray import Document, DocumentArray
import numpy as np

n_dim = 3
metric = 'Euclidean'

da = DocumentArray(
    storage='annlite',
    config={'n_dim': n_dim, 'columns': [('price', 'float')], 'metric': metric},
)

with da:
    da.extend(
        [
            Document(id=f'r{i}', embedding=i * np.ones(n_dim), tags={'price': i})
            for i in range(10)
        ]
    )

max_price, n_limit = 7, 4
np_query = np.ones(n_dim) * 8
filter = {'price': {'$lte': max_price}}

results = da.find(np_query, filter=filter, limit=n_limit)
results[:, 'embedding']
```

```text
array([[7., 7., 7.],
       [6., 6., 6.],
       [5., 5., 5.],
       [4., 4., 4.]])
```

## Known limitations


### Multiple references to the same storage backend

Let's see an example with ANNLite storage backend, other storage backends would also have the same problem. Let's create two DocumentArrays `da` and `db` that point the same storage backend:

```python
from docarray import DocumentArray, Document

da = DocumentArray(storage='annlite', config={'data_path': './temp3', 'n_dim': 2})
da.append(Document(text='hello'))
print(len(da))

db = DocumentArray(storage='annlite', config={'data_path': './temp3', 'n_dim': 2})
print(len(db))
```

The output is:
```text
1
0
```

Looks like `db` is not really up-to-date with `da`. This is true and false. True in the sense that `1` is not `0`, number speaks by itself. 
False in the sense that, the Document is already written to the storage backend. You just can't see it. 

To prove it does persist, run the following code snippet multiple times and you will see the length is increasing one at a time:

```python
from docarray import DocumentArray, Document

da = DocumentArray(storage='annlite', config={'data_path': './temp3', 'n_dim': 2})
da.append(Document(text='hello'))
print(len(da))
```

Simply put, the reason of this behavior is that certain meta information **not synced immediately** to the backend on *every* operation; it would be very costly to do so.
As a consequence, your multiple references to the same backend  would look different if they are written in one code block as the example above.

To solve this problem, simply use `with` statement and use DocumentArray as a context manager. The last example can be refactored into the following: 

```{code-block} python
---
emphasize-lines: 4,5
---
from docarray import DocumentArray, Document

da = DocumentArray(storage='annlite', config={'data_path': './temp4', 'n_dim': 2})
with da:
    da.append(Document(text='hello'))
print(len(da))

db = DocumentArray(storage='annlite', config={'data_path': './temp4', 'n_dim': 2})
print(len(db))
```

Now you get the correct output:
```text
1
1
```

Take home message is, use the context manager and put your write operations into the `with` block, when you work with multiple references in a row.

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
