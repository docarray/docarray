# Milvus Document Index

!!! note "Install dependencies"
    To use [MilvusDocumentIndex][docarray.index.backends.milvus.MilvusDocumentIndex], you need to install extra dependencies with the following command:

    ```console
    pip install "docarray[milvus]"
    ```

This is the user guide for the [MilvusDocumentIndex][docarray.index.backends.milvus.MilvusDocumentIndex],
focusing on special features and configurations of Redis.


## Basic Usage
```python
from docarray import BaseDoc, DocList
from docarray.index import MilvusDocumentIndex
from docarray.typing import NdArray
from pydantic import Field
import numpy as np

# Define the document schema.
class MyDoc(BaseDoc):
    title: str 
    embedding: NdArray[128] = Field(is_embedding=True)

# Create dummy documents.
docs = DocList[MyDoc](MyDoc(title=f'title #{i}', embedding=np.random.rand(128)) for i in range(10))

# Initialize a new MilvusDocumentIndex instance and add the documents to the index.
doc_index = MilvusDocumentIndex[MyDoc](host='localhost')
doc_index.index(docs)

# Perform a vector search.
query = np.ones(128)
retrieved_docs = doc_index.find(query, limit=10)
```


