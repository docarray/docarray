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

```python
from pydantic import Field

from docarray import BaseDoc
from docarray.index import HnswDocumentIndex
from docarray.typing import NdArray


class SimpleSchema(BaseDoc):
    tens: NdArray[10] = Field(dim=128, space='cosine')


doc_index = HnswDocumentIndex[SimpleSchema](work_dir='./tmp')
```

### Index

## ElasticSearch
