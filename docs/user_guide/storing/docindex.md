# Introduction

A Document Index lets you store your documents and search through them using vector similarity.

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
- [Epsilla](https://epsilla.com/)  |  [Docs](index_epsilla.md)
- [Redis](https://redis.com/)  |  [Docs](index_redis.md)
- [Milvus](https://milvus.io/)  |  [Docs](index_milvus.md)
- [HNSWlib](https://github.com/nmslib/hnswlib)  |  [Docs](index_hnswlib.md)
- InMemoryExactNNIndex  |  [Docs](index_in_memory.md)


## Basic usage

Let's learn the basic capabilities of Document Index with [InMemoryExactNNIndex][docarray.index.backends.in_memory.InMemoryExactNNIndex]. 
This doesn't require a database server - rather, it saves your data locally.


!!! note "Using a different vector database"
    You can easily use Weaviate, Qdrant, Redis, Milvus or Elasticsearch instead -- their APIs are largely identical!
    To do so, check their respective documentation sections.

!!! note "InMemoryExactNNIndex in more detail"
    The following section only covers the basics of InMemoryExactNNIndex. 
    For a deeper understanding, please look into its [documentation](index_in_memory.md).

### Define document schema and create data
The following code snippet defines a document schema using the `BaseDoc` class. Each document consists of a title (a string), 
a price (an integer), and an embedding (a 128-dimensional array). It also creates a list of ten documents with dummy titles, 
prices ranging from 0 to 9, and randomly generated embeddings.
```python
from docarray import BaseDoc, DocList
from docarray.index import InMemoryExactNNIndex
from docarray.typing import NdArray
import numpy as np


class MyDoc(BaseDoc):
    title: str
    price: int
    embedding: NdArray[128]


docs = DocList[MyDoc](
    MyDoc(title=f"title #{i}", price=i, embedding=np.random.rand(128))
    for i in range(10)
)
```

### Initialize the Document Index and add data
Here we initialize an `InMemoryExactNNIndex` instance with the document schema we defined previously, and add the created documents to this index.
```python
doc_index = InMemoryExactNNIndex[MyDoc]()
doc_index.index(docs)
```

### Perform a vector similarity search
Now, let's perform a similarity search on the document embeddings. 
As a result, we'll retrieve the ten most similar documents and their corresponding similarity scores.
```python
query = np.ones(128)
retrieved_docs, scores = doc_index.find(query, search_field='embedding', limit=10)
```

### Filter documents
In this snippet, we filter the indexed documents based on their price field, specifically retrieving documents with a price less than 5:
```python
query = {'price': {'$lt': 5}}
filtered_docs = doc_index.filter(query, limit=10)
```

### Combine different search methods
The final snippet combines the vector similarity search and filtering operations into a single query. 
We first perform a similarity search on the document embeddings and then apply a filter to return only those documents with a price greater than or equal to 2:
```python
query = (
    doc_index.build_query()  # get empty query object
    .find(query=np.ones(128), search_field='embedding')  # add vector similarity search
    .filter(filter_query={'price': {'$gte': 2}})  # add filter search
    .build()  # build the query
)
retrieved_docs, scores = doc_index.execute_query(query)
```

## Learn more
The code snippets above just scratch the surface of what a Document Index can do. 
To learn more and get the most out of `DocArray`, take a look at the detailed guides for the vector database backends you're interested in:

- [Weaviate](https://weaviate.io/)  |  [Docs](index_weaviate.md)
- [Qdrant](https://qdrant.tech/)  |  [Docs](index_qdrant.md)
- [Elasticsearch](https://www.elastic.co/elasticsearch/) v7 and v8  |  [Docs](index_elastic.md)
- [Epsilla](https://epsilla.com/)  |  [Docs](index_epsilla.md)
- [Redis](https://redis.com/)  |  [Docs](index_redis.md)
- [Milvus](https://milvus.io/)  |  [Docs](index_milvus.md)
- [HNSWlib](https://github.com/nmslib/hnswlib)  |  [Docs](index_hnswlib.md)
- InMemoryExactNNIndex  |  [Docs](index_in_memory.md)
