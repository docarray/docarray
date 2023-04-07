# Index
This section show you how to use the `DocArray.index` module. `DocArray.index` module is used to create index for the tensors so that one can search the document based on the vector similarity. `DocArray.index` implements the following index.

## Hnswlib

[HnswDocumentIndex][docarray.index.backends.hnswlib.HnswDocumentIndex] implement the index based on [hnswlib](https://github.com/nmslib/hnswlib). This is a lightweight implementation with vectors stored in memory.

!!! note
    To use [HnswDocumentIndex][docarray.index.backends.hnswlib.HnswDocumentIndex], one need to install the extra dependency with the following command

    ```console
    pip install "docarray[hnswlib]"
    ```

### Construct
To construct an index, you need to define the schema first. You can define the schema in the same way as define a `Doc`. The only difference is that you need to define the dimensionality of the vector space by `dim` and the name of the space by `space`. The `dim` argument must be an integer. The `space` argument can be one of `l2`, `ip` or `cosine`. TODO: add links to the detailed explaination

`work_dir` is the directory for storing the index. If there is an index in the directory, it will be automatically loaded. When the schema of the saved and the defined index do not match, an exception will be raised.

```python
from pydantic import Field

from docarray import BaseDoc
from docarray.index import HnswDocumentIndex
from docarray.typing import NdArray


class SimpleSchema(BaseDoc):
    tensor: NdArray[128] = Field(dim=128, space='cosine')


doc_index = HnswDocumentIndex[SimpleSchema](work_dir='./tmp')
```

### Index
Use `.index()` to add `Doc` into the index. You need to define the `Doc` following the schema of the index. `.num_docs()` returns the total number of `Doc` in the index.

```python
from docarray import BaseDoc
from docarray.typing import NdArray
import numpy as np

class SimpleDoc(BaseDoc):
    tensor: NdArray[128]

index_docs = [SimpleDoc(tensor=np.zeros(128)) for _ in range(64)]

doc_index.index(index_docs)
print(f'number of docs in the index: {doc_index.num_docs()}')
```

### Access
To access the `Doc`, you need to specify the `id`. You can also pass a list of `id` to access multiple `Doc`.

```python
# access a single Doc
doc_index[index_docs[16].id]

# access multiple Docs
doc_index[index_docs[16].id, index_docs[17].id]
```

### Delete
To delete the `Doc`, use the built-in function `del` with the `id` of the `Doc` to be deleted. You can also pass a list of `id` to delete multiple `Doc`.

```python
# delete a single Doc
del doc_index[index_docs[16].id]

# delete multiple Docs
del doc_index[index_docs[16].id, index_docs[17].id]
```

### Find nearest neighbors
Use `.find()` to find the nearest neighbors. You can use `limit` argument to configurate how much `Doc` to return.

```python
query = SimpleDoc(tensor=np.ones(10))

docs, scores = doc_index.find(query, limit=5)
```

### Nested index
When using the index, you can define multiple fields as well as the nested structure. 

```python
# example of construct nested and flat index
```

Use the `search_field` to specify which field to be used when performing the vector search. You can use the dunder operator to specify the field defined in the nested data.

```python
# example of find nested and flat index
```

To delete a nested data, ...

```python
# example of delete nested and flat index
```