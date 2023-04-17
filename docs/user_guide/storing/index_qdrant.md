# Qdrant Document Index

!!! note "Install dependencies"
    To use [QdrantDocumentIndex][docarray.index.backends.qdrant.QdrantDocumentIndex], you need to install extra dependencies with the following command:

    ```console
    pip install "docarray[qdrant]"
    ```

The following is a starter script for using the [QdrantDocumentIndex][docarray.index.backends.qdrant.QdrantDocumentIndex],
based on the [Qdrant](https://qdrant.tech/) vector search engine.

!!! tip "See all configuration options"
    To see all configuration options for the [QdrantDocumentIndex][docarray.index.backends.qdrant.QdrantDocumentIndex],
    you can do the following:

    ```python
    from docarray.index import QdrantDocumentIndex

    # the following can be passed to the __init__() method
    db_config = QdrantDocumentIndex.DBConfig()
    print(db_config)  # shows default values

    # the following can be passed to the configure() method
    runtime_config = QdrantDocumentIndex.RuntimeConfig()
    print(runtime_config)  # shows default values
    ```

```python
import numpy as np

from typing import Optional

from docarray import BaseDoc
from docarray.index import QdrantDocumentIndex
from docarray.typing import NdArray

from qdrant_client.http import models


class MyDocument(BaseDoc):
    title: str
    title_embedding: NdArray[786]
    image_path: Optional[str]
    image_embedding: NdArray[512]


# Creating an in-memory Qdrant document index
qdrant_config = QdrantDocumentIndex.DBConfig(":memory:")
doc_index = QdrantDocumentIndex[MyDocument](qdrant_config)

# Indexing the documents
doc_index.index(
    [
        MyDocument(
            title=f"My document {i}",
            title_embedding=np.random.random(786),
            image_path=None,
            image_embedding=np.random.random(512),
        )
        for i in range(100)
    ]
)

# Performing a vector search only
results = doc_index.find(
    query=np.random.random(512),
    search_field="image_embedding",
    limit=3,
)

# Connecting to a local Qdrant instance with Scalar Quantization enabled,
# and using non-default collection name to store the datapoints
qdrant_config = QdrantDocumentIndex.DBConfig(
    "http://localhost:6333",
    collection_name="another_collection",
    quantization_config=models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8,
            quantile=0.99,
            always_ram=True,
        ),
    ),
)
doc_index = QdrantDocumentIndex[MyDocument](qdrant_config)

# Indexing the documents
doc_index.index(
    [
        MyDocument(
            title=f"My document {i}",
            title_embedding=np.random.random(786),
            image_path=None,
            image_embedding=np.random.random(512),
        )
        for i in range(100)
    ]
)

# Text lookup, without vector search. Using the Qdrant filtering mechanisms:
# https://qdrant.tech/documentation/filtering/
results = doc_index.filter(
    filter_query=models.Filter(
        must=[
            models.FieldCondition(
                key="title",
                match=models.MatchText(text="document 2"),
            ),
        ],
    ),
)
```