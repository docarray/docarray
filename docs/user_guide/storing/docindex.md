# Introduction

A Document Index lets you store your Documents and search through them using vector similarity.

This is useful if you want to store a bunch of data, and at a later point retrieve documents that are similar to
some query that you provide.
Relevant concrete examples are neural search applications, augmenting LLMs and chatbots with domain knowledge ([Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)),
or recommender systems.

!!! question "How does vector similarity search work?"
    Without going into too much detail, the idea behind vector similarity search is the following:

    You represent every data point that you have (in our case, a document) as a _vector_, or _embedding_.
    This vector should represent as much semantic information about your data as possible: Similar data points should
    be represented by similar vectors.

    These vectors (embeddings) are usually obtained by passing the data through a suitable neural network that has been
    trained to produce such semantic representations - this is the _encoding_ step.

    Once you have your vectors that represent your data, you can store them, for example in a vector database.
    
    To perform similarity search, you take your input query and encode it in the same way as the data in your database.
    Then, the database will search through the stored vectors and return those that are most similar to your query.
    This similarity is measured by a _similarity metric_, which can be [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity),
    [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance), or any other metric that you can think of.

    If you store a lot of data, performing this similarity computation for every data point in your database is expensive.
    Therefore, vector databases usually perform _approximate nearest neighbor (ANN)_ search.
    There are various algorithms for doing this, such as [HNSW](https://arxiv.org/abs/1603.09320), but in a nutshell,
    they allow you to search through a large database of vectors very quickly, at the expense of a small loss in accuracy.

DocArray's Document Index concept achieves this by providing a unified interface to a number of [vector databases](https://learn.microsoft.com/en-us/semantic-kernel/concepts-ai/vectordb).
In fact, you can think of Document Index as an **[ORM](https://sqlmodel.tiangolo.com/db-to-code/) for vector databases**.

Currently, DocArray supports the following vector databases:

- [Weaviate](https://weaviate.io/)  |  [Docs](index_weaviate.md)
- [Qdrant](https://qdrant.tech/)  |  [Docs](index_qdrant.md)
- [Elasticsearch](https://www.elastic.co/elasticsearch/) v7 and v8  |  [Docs](index_elastic.md)
- [HNSWlib](https://github.com/nmslib/hnswlib)  |  [Docs](index_hnswlib.md)
- InMemoryExactNNSearch  |  [Docs](index_in_memory.md)


## Basic Usage

For this user guide you will use the [InMemoryExactNNSearch][docarray.index.backends.in_memory.InMemoryExactNNSearch]
because it doesn't require you to launch a database server. Instead, it will store your data locally.

!!! note "Using a different vector database"
    You can easily use Weaviate, Qdrant, or Elasticsearch instead -- they share the same API!
    To do so, check their respective documentation sections.

!!! note "InMemory-specific settings"
    The following sections explain the general concept of Document Index by using
    [InMemoryExactNNSearch][docarray.index.backends.in_memory.InMemoryExactNNSearch] as an example.
    For InMemory-specific settings, check out the [InMemoryExactNNSearch][docarray.index.backends.in_memory.InMemoryExactNNSearch] documentation
    [here](index_in_memory.md).


```python
from docarray import BaseDoc, DocList
from docarray.index import HnswDocumentIndex
from docarray.typing import NdArray
import numpy as np

# Define the document schema.
class MyDoc(BaseDoc):
    title: str
    price: int
    embedding: NdArray[128]

# Create documents (using dummy/random vectors)
docs = DocList[MyDoc](MyDoc(title=f'title #{i}', price=i, embedding=np.random.rand(128)) for i in range(10))

# Initialize a new HnswDocumentIndex instance and add the documents to the index.
doc_index = HnswDocumentIndex[MyDoc](workdir='./my_index')
doc_index.index(docs)

# Perform a vector search.
query = np.ones(128)
retrieved_docs = doc_index.find(query, search_field='embedding', limit=10)

# Perform filtering (price < 5)
query = {'price': {'$lt': 5}}
filtered_docs = doc_index.filter(query, limit=10)

# Perform a hybrid search - combining vector search with filtering
query = (
    doc_index.build_query()  # get empty query object
    .find(np.ones(128), search_field='embedding')  # add vector similarity search
    .filter(filter_query={'price': {'$gte': 2}})  # add filter search
    .build()  # build the query
)
results = doc_index.execute_query(query)
```
