# Migration guide

If you are using DocArray v<0.30.0, you will be familiar with its [dataclass API](https://docarray.jina.ai/fundamentals/dataclass/).

_DocArray v2 is that idea, taken seriously._ Every document is created through dataclass-like interface,
courtesy of [Pydantic](https://pydantic-docs.helpmanual.io/usage/models/).

This gives the following advantages:

- **Flexibility:** No need to conform to a fixed set of fields -- your data defines the schema
- **Multi-modality:** Easily store multiple modalities and multiple embeddings in the same Document
- **Language agnostic:** At its core, Documents are just dictionaries. This makes it easy to create and send them from any language, not just Python.

You may also be familiar with our old Document Stores for vector DB integration.
They are now called **Document Indexes** and offer the following improvements:

- **Hybrid search:** You can now combine vector search with text search, and even filter by arbitrary fields
- **Production-ready:** The new Document Indexes are a much thinner wrapper around the various vector DB libraries, making them more robust and easier to maintain
- **Increased flexibility:** We strive to support any configuration or setting that you could perform through the DB's first-party client

For now, Document Indexes support **[Weaviate](https://weaviate.io/)**, **[Qdrant](https://qdrant.tech/)**, **[ElasticSearch](https://www.elastic.co/)**, and **[HNSWLib](https://github.com/nmslib/hnswlib)**, with more to come.

## Changes to `Document`

- `Document` has been renamed to [`BaseDoc`][docarray.BaseDoc].
- `BaseDoc` can not be used directly, but instead has to be extended. Therefore, each document class is created through a dataclass-like interface.
- Following from the previous point, the extending of `BaseDoc` allows for a flexible schema while the 
`Document` class in v1 only allowed for a fixed schema, with one of `tensor`, `text` and `blob`, 
and additional `chunks` and `matches`.
- Due to the added flexibility, one can not know what fields your document class will provide. 
  Therefore, various methods from v1 (such as `.load_uri_to_image_tensor()`) are not supported in v2.
  Instead, we provide some of those methods on [typing-level](data_types/first_steps.md). 
- In v2 we have the [`LegacyDocument`][docarray.documents.legacy.LegacyDocument] class, 
  which extends `BaseDoc` while following the same schema as v1's `Document`.
  The `LegacyDocument` can be useful to start migrating your codebase from v1 to v2. 
  Nevertheless, the API is not fully compatible with DocArray v1 `Document`.
  Indeed, none of the method associated with `Document` are present. 
  Only the schema of the data is similar.

## Changes to `DocumentArray`

### DocList

- The `DocumentArray` class from v1 has been renamed to [`DocList`][docarray.array.DocList], 
to be more descriptive of its actual functionality, since it is a list of `BaseDoc`s

### DocVec

- Additionally, we introduced the class [`DocVec`][docarray.array.DocVec], which is a column based representation of `BaseDoc`s. 
Both `DocVec` and `DocList` extend `AnyDocArray`.
- `DocVec` is a container of Documents appropriates to perform computation that require batches of data 
(ex: matrix multiplication, distance calculation, deep learning forward pass).
- A `DocVec` has a similar interface as `DocList`
but with an underlying implementation that is column based instead of row based.
Each field of the schema of the `DocVec` (the `.doc_type` which is a
`BaseDoc`) will be stored in a column.
If the field is a tensor, the data from all Documents will be stored as a single
doc_vec (torch/np/tf) tensor. If the tensor field is `AnyTensor` or a Union of tensor types, the
`.tensor_type` will be used to determine the type of the doc_vec column. 

### Parameterized DocList
- With the added flexibility of your document schema, and therefore endless options to design your document schema, 
when initializing a `DocList` it does not necessarily have to be homogenous. 
- If you want a homogenous `DocList` you can parameterize it at initialization time:
```python
from docarray import DocList
from docarray.documents import ImageDoc

docs = DocList[ImageDoc]()
```

- Methods like `.from_csv()` or `.pull()` only work with parameterized `DocList`s. 

### Access attributes of your DocumentArray

- In v1 you could access an attribute of all Documents in your DocumentArray by calling the plural 
of the attribute's name on your DocArray instance. 
- In v2 you don't have to use the plural, but instead just use the document's attribute name, 
since `AnyDocArray` will expose the same attributes as the `BaseDoc`s it contains.
This will return a list of `type(attribute)`.
However, this only works if (and only if) all the `BaseDoc`s in the `AnyDocArray` have the same schema. Therfore this only works

```python
from docarray import BaseDoc, DocList


class Book(BaseDoc):
    title: str
    author: str = None


docs = DocList[Book]([Book(title=f'title {i}') for i in range(5)])
book_titles = docs.title  # returns a list[str]

# this would fail
# docs = DocList([Book(title=f'title {i}') for i in range(5)])
# book_titles = docs.title
```

## Changes to Document Store

In v2 the `Document Store` has been renamed to [`DocIndex`](user_guide/storing/first_steps.md) and can be used for fast retrieval using vector similarity. 
DocArray v2 `DocIndex` supports:

- [Weaviate](https://weaviate.io/)
- [Qdrant](https://qdrant.tech/)
- [ElasticSearch](https://www.elastic.co/)
- [HNSWLib](https://github.com/nmslib/hnswlib)

Instead of creating a `DocumentArray` instance and setting the `storage` parameter to a vector database of your choice, 
in v2 you can initialize a `DocIndex` object of your choice, such as: 

```python
db = HnswDocumentIndex[MyDoc](work_dir='/my/work/dir')
```

In contrast, [`DocStore`](user_guide/storing/first_step.md#document-store) in v2 can be used for simple long-term storage, such as with AWS S3 buckets or JINA AI Cloud.