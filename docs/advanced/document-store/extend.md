# Add New Document Store

DocumentArray can be easily extended to support new Document Store. As we have seen in the previous chapters, a document store can be a SQL/NoSQL/vector database, or even an in-memory data structure. 

For DocArray, the motivation of on-boarding a new store is often:
- having persistence that better fits to the use case;
- pulling from an existing data source;
- supporting advanced query languages, e.g. nearest-neighbor retrieval.

For the database vendor, the motivation is often:
- having a powerful, well-designed and well-maintained Python client for your document store;
- plugging your document store into Jina AI ecosystems (e.g. Jina, Hub, CLIP-as-service, Finetuner, etc.) and making synergy with Jina AI.

After the extension, users can enjoy convenient and powerful DocumentArray API on top of your document store. It promises the same user experience just like using a regular DocumentArray, no extra learning is required.

This chapter gives you a walk-through on how to add a new document store. To be specific, in this chapter we are extending DocumentArray to support a new document store called `mydocstore`. The final usage would look like the following:

```python
from docarray import DocumentArray

da = DocumentArray(storage='mydocstore', config={...})
```

Let's get started!

## Step 1: create the folder

Go to `docarray/array/storage` folder, create a sub-folder for your document store. Let's call it `mydocstore`. You need to create four empty files in that folder:

```{code-block} 
---
emphasize-lines: 8-13
---
README.md
docarray
    |
    |--- array
            |
            |--- storage
                    |
                    |--- mydocstore
                            |
                            |--- __init__.py
                            |--- getsetdel.py
                            |--- seqlike.py
                            |--- backend.py
```

These four files consist of necessary interface for making the extension work on DocumentArray. Additionally, if your 
storage backend supports approximate nearest-neighbor search, you can include another file 'find.py'.

## Step 2: implement `getsetdel.py` 

Your `getsetdel.py` should look like the following:

```python
from docarray.array.storage.base.getsetdel import BaseGetSetDelMixin
from docarray import Document


class GetSetDelMixin(BaseGetSetDelMixin):
    def _get_doc_by_id(self, _id: str) -> 'Document':
        # to be implemented
        ...

    def _del_doc_by_id(self, _id: str):
        # to be implemented
        ...

    def _set_doc_by_id(self, _id: str, value: 'Document'):
        # to be implemented
        ...

    def _load_offset2ids(self):
        # to be implemented
        ...

    def _save_offset2ids(self):
        # to be implemented
        ...
```

You need to implement the above five functions, which correspond to the logics of get/set/delete items via a string `.id`. They are essential to ensure DocumentArray works.

Note that DocumentArray maintains an `offset2ids` mapping to allow a list-like behaviour. This mapping is 
inherited from the `BaseGetSetDelMixin`. Therefore, you need to implement methods to persist this mapping, in case you 
want to also persist the ordering of Documents inside the storage. However, the list-like
structure implemented by Offset2id can introduce performance bottlenecks, so you can disable this feature by passing a flag when constructing the backend.

In step 4 you will see how you should read in this flag (`list_like`) from the user. Here you have to use it do adapt some operations.

In your backend implementations, in `getsetdel.py`, you have to construct the _offset2ids 
member variable by passing list_like flag as follows:

```Python
    def _load_offset2ids(self):
        if self._list_like:
            ids = self._get_offset2ids_meta()
            self._offset2ids = Offset2ID(ids, list_like=self._list_like)
        else:
            self._offset2ids = Offset2ID([], list_like=self._list_like)
```

Note that this flag should be stored in `self._list_like`, so that other parts of the DocumentArray implementation can leverage it.

Keep in mind that `_del_doc_by_id` and `_set_doc_by_id` **must not** update `offset2ids`, we handle that for you in an 
upper level. Also, make sure that `_set_doc_by_id` performs an **upsert operation** and removes the old ID (`_id`) in case 
`value.id` is different from `_id`.


```{tip}
Let's call the above five functions as **the essentials**.

If you aim for high performance, it is recommeneded to implement other methods *without* leveraging your essentials. They are: `_get_docs_by_ids`, `_del_docs_by_ids`, `_clear_storage`, `_set_doc_value_pairs`, `_set_doc_value_pairs_nested`, `_set_docs_by_ids`. You can get their full signatures from {class}`~docarray.array.storage.base.getsetdel.BaseGetSetDelMixin`. These functions define more fine-grained get/set/delete logics that are frequently used in DocumentArray. 

Implementing them is fully optional, and you can only implement some of them not all of them. If you are not implementing them, those methods use a generic-but-slow version based on your five essentials.
```

```{seealso}
As a reference, you can check out how we implement for SQLite, check out {class}`~docarray.array.storage.sqlite.getsetdel.GetSetDelMixin`.
```

## Step 3: implement `seqlike.py`

Your `seqlike.py` should look like the following:

```python
from typing import Iterable, Iterator, Union, TYPE_CHECKING
from docarray.array.storage.base.seqlike import BaseSequenceLikeMixin

if TYPE_CHECKING:
    from docarray import Document


class SequenceLikeMixin(BaseSequenceLikeMixin):
    def __eq__(self, other):
        ...

    def __contains__(self, x: Union[str, 'Document']):
        ...

    def __repr__(self):
        ...

    def __add__(self, other: Union['Document', Iterable['Document']]):
        ...

    def __len__(self):
        ...

    def insert(self, index: int, value: 'Document'):
        # Optional. By default, this adds a new item and update offset2id
        # if you want to customize this, make sure to handle offset2id
        ...

    def _append(self, value: 'Document'):
        # Optional. Override this if you have a better implementation than inserting at the last position
        ...

    def _extend(self, values: Iterable['Document']) -> None:
        # Optional. Override this if you have better implementation than appending one by one
        ...

    def __iter__(self) -> Iterator['Document']:
        # Optional. By default, this relies on offset2id to iterate
        ...
```

Most of the interfaces come from Python standard [MutableSequence](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableSequence).

```{seealso}
As a reference, to see how we implement for SQLite, check out {class}`~docarray.array.storage.sqlite.seqlike.SequenceLikeMixin`.
```

To support the list-like feature, the list-like APIs should perform flag checking only when the offset2id structure is called as follows:
```python
def _extend(self, docs: Iterable['Document']):
    da = DocumentArray(docs)
    for batch_of_docs in da.batch(self._config.batch_size):
        self._upload_batch(batch_of_docs)
        if self._list_like:
            self._offset2ids.extend(batch_of_docs[:, 'id'])
```

## Step 4: implement `backend.py`

Your `backend.py` should look like the following:

```python
from typing import Optional, TYPE_CHECKING, Union, Dict
from dataclasses import dataclass

from docarray.array.storage.base.backend import BaseBackendMixin

if TYPE_CHECKING:
    from docarray.typing import (
        DocumentArraySourceType,
    )


@dataclass
class MyDocStoreConfig:
    config1: str
    config2: str
    config3: Dict
    ...


class BackendMixin(BaseBackendMixin):
    def _init_storage(
        self,
        _docs: Optional['DocumentArraySourceType'] = None,
        config: Optional[Union[MyDocStoreConfig, Dict]] = None,
        **kwargs
    ):
        super()._init_storage(_docs, config, **kwargs)
        ...

    def _ensure_unique_config(
        self,
        config_root: dict,
        config_subindex: dict,
        config_joined: dict,
        subindex_name: str,
    ) -> dict:
        ...  # ensure unique identifiers here
        return config_joined
```

`MyDocStoreConfig` is a dataclass for containing the configs. You can expose arguments of your document store to this data class and allow users to customize them. In `init_storage` function, you need to parse `config` either from `MyDocStoreConfig` object or a `Dict`.

To allow the disabling of list-like features, your configuration should accept the flag `list_like` as follows:
```python
@dataclass
class MyDocStoreConfig:
    config1: str
    config2: str
    list_like: bool
    config3: Dict
    ...
```
By default, this should be set to `True`.

Further, you have to store the value of this flag in `self._list_like`. Some methods that are handled outside of your control will take the value form there and use it appropriately.

`_init_storage` is a very important function to be called during the DocumentArray construction.
You need to handle different construction and copy behaviors in this function.

`_ensure_unique_config` is needed to support DocArray's subindex feature.
A subindex inherits its configuration from the root index, unless a field of the configuration is explicitly provided to the subindex.
Usually however, each table in a database has to have a unique identifier (e.g. 'name', 'table_name', 'data_path', etc.).
In order to avoid clashes you need to make sure that this identifier is actually unique between parent und subindices, despite the inheritance of configurations.


```{seealso}
As a reference, you can check out how we implement for SQLite here: {class}`~docarray.array.storage.sqlite.backend.BackendMixin`.
```

## Step 5: implement `find.py`
If your storage backend supports approximate nearest neighbor search, you can allow users to use this feature within 
docarray. To do so, add a `find.py` file that looks like the following:

```python
from typing import TYPE_CHECKING, TypeVar, List, Union

if TYPE_CHECKING:
    import numpy as np

    # Define the expected input type that your ANN search supports
    MyDocumentStoreArrayType = TypeVar('MyDocumentStoreArrayType', np.ndarray, ...)


class FindMixin:
    def _find_similar_vectors(
        self, query: 'MyDocumentStoreArrayType', limit=10
    ) -> 'DocumentArray':
        """Expects a MyDocumentStoreArrayType vector query and should return a DocumentArray of results retrieved from
        the storage backend"""
        ...

    def _find(
        self, query: 'ElasticArrayType', limit: int = 10, **kwargs
    ) -> Union['DocumentArray', List['DocumentArray']]:
        """Returns `limit` approximate nearest neighbors given a batch of input queries.
        If the query is a single query, should return a DocumentArray, otherwise a list of DocumentArrays containing
        the closest Documents for each query.
        """
        ...
```

Make sure to store the distance scores in the `.scores` dictionary of the Documents that are being returned with the `distance` value as key.

## Step 6: summarize everything in `__init__.py`.

Your `__init__.py` should look like the following:

```python
from abc import ABC

from .backend import BackendMixin, MyDocStoreConfig
from .getsetdel import GetSetDelMixin
from .seqlike import SequenceLikeMixin

__all__ = ['StorageMixins', 'MyDocStoreConfig']


class StorageMixins(BackendMixin, GetSetDelMixin, SequenceLikeMixin, ABC):
    ...
```

Just copying and pasting it should work.

If you have implemented a `find.py` module, make sure to also inherit the `FindMixin`:
```python
class StorageMixins(FindMixin, BackendMixin, GetSetDelMixin, SequenceLikeMixin, ABC):
    ...
```

## Step 7: subclass from `DocumentArray`

Create a file `mydocstore.py` under `docarray/array/`

```{code-block}
---
emphasize-lines: 6
---
README.md
docarray
    |
    |--- array
            |
            |--- mydocstore.py
            |--- storage
                    |
                    |--- mydocstore
                            |
                            |--- __init__.py
                            |--- getsetdel.py
                            |--- seqlike.py
                            |--- backend.py
```

The file content should look like the following:

```python
from .document import DocumentArray

from .storage.mydocstore import StorageMixins, MyDocStoreConfig

__all__ = ['MyDocStoreConfig', 'DocumentArrayMyDocStore']


class DocumentArrayMyDocStore(StorageMixins, DocumentArray):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
```


## Step 8: add entrypoint to `DocumentArray`

We are almost there! Now we need to add the entrypoint to `DocumentArray` constructor to allow user to use the `mydocstore` backend as follows:

```python
from docarray import DocumentArray

da = DocumentArray(storage='mydocstore')
```

Go to `docarray/array/document.py` and add `mydocstore` there:

```{code-block} python
---
emphasize-lines: 7-10
---
class DocumentArray(AllMixins, BaseDocumentArray):
    
    ...
    
    def __new__(cls, *args, storage: str = 'memory', **kwargs) -> 'DocumentArrayLike':
        if cls is DocumentArray:
            if storage == 'mydocstore':
                from .mydocstore import DocumentArrayMyDocStore

                instance = super().__new__(DocumentArrayMyDocStore)
            elif storage == 'memory':
                from .memory import DocumentArrayInMemory
                ...  
```

Done! Now you should be able to use it like `DocumentArrayMyDocStore`!

## On pull request: add tests and type-hint

You are welcome to contribute your extension back to DocArray. You need to include `DocumentArrayMyDocStore` in at least the following tests:

```text
tests/unit/array/test_advance_indexing.py
tests/unit/array/test_sequence.py
tests/unit/array/test_construct.py
```

Please also add `@overload` type hint to `docarray/array/document.py`.

```python
class DocumentArray(AllMixins, BaseDocumentArray):
    ...

    @overload
    def __new__(
        cls,
        _docs: Optional['DocumentArraySourceType'] = None,
        storage: str = 'mydocstore',
        config: Optional[Union['MyDocStoreConfig', Dict]] = None,
    ) -> 'DocumentArrayMyDocStore':
        """Create a MyDocStore-powered DocumentArray object."""
        ...
```

Now you are ready to commit the contribution and open a pull request. 
