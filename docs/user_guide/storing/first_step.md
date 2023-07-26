# Introduction

In the previous sections we saw how to use [`BaseDoc`][docarray.base_doc.doc.BaseDoc], [`DocList`][docarray.array.doc_list.doc_list.DocList] and [`DocVec`][docarray.array.doc_vec.doc_vec.DocVec] to represent multimodal data and send it over the wire.
In this section we will see how to store and persist this data.

DocArray offers two ways of storing your data, each of which have their own documentation sections:

1. **[Document Store](#document-store)** for simple long-term storage
2. **[Document Index](#document-index)** for fast retrieval using vector similarity

## Document Store
    
[DocList][docarray.array.doc_list.doc_list.DocList] can be persisted using the
[`.push()`][docarray.array.doc_list.pushpull.PushPullMixin.push] and 
[`.pull()`][docarray.array.doc_list.pushpull.PushPullMixin.pull] methods. 
Under the hood, [DocStore][docarray.store.abstract_doc_store.AbstractDocStore] is used to persist a `DocList`. 
You can either store your documents on-disk or upload them to [AWS S3](https://aws.amazon.com/s3/), 
[minio](https://min.io) or [Jina AI Cloud](https://cloud.jina.ai/user/storage). 

This section covers the following three topics:

  - [Storing](doc_store/store_file.md) [`BaseDoc`][docarray.base_doc.doc.BaseDoc], [`DocList`][docarray.array.doc_list.doc_list.DocList] and [`DocVec`][docarray.array.doc_vec.doc_vec.DocVec] on-disk
  - [Storing on Jina AI Cloud](doc_store/store_jac.md) 
  - [Storing on S3](doc_store/store_s3.md)
   
## Document Index

A Document Index lets you store your Documents and search through them using vector similarity.

This is useful if you want to store a bunch of data, and at a later point retrieve documents that are similar to
a query that you provide.
Relevant concrete examples are neural search applications, augmenting LLMs and chatbots with domain knowledge ([Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401))]),
or recommender systems.

DocArray's Document Index concept achieves this by providing a unified interface to a number of [vector databases](https://learn.microsoft.com/en-us/semantic-kernel/concepts-ai/vectordb).
In fact, you can think of Document Index as an **[ORM](https://sqlmodel.tiangolo.com/db-to-code/) for vector databases**.

Currently, DocArray supports the following vector indexes. Some of them wrap vector databases (Weaviate, Qdrant, ElasticSearch) and act as a client for them, while others
use a vector search library locally (HNSWLib, Exact NN search):

- [Weaviate](https://weaviate.io/)  |  [Docs](index_weaviate.md)
- [Qdrant](https://qdrant.tech/)  |  [Docs](index_qdrant.md)
- [Elasticsearch](https://www.elastic.co/elasticsearch/) v7 and v8  |  [Docs](index_elastic.md)
- [Redis](https://redis.com/)  |  [Docs](index_redis.md)
- [Milvus](https://milvus.io/)  |  [Docs](index_milvus.md)
- [Hnswlib](https://github.com/nmslib/hnswlib)  |  [Docs](index_hnswlib.md)
- InMemoryExactNNSearch  |  [Docs](index_in_memory.md)
