# Intro

In the previous sections we saw how to use [`BaseDoc`][docarray.base_doc.doc.BaseDoc], [`DocList`][docarray.array.doc_list.doc_list.DocList] and [`DocVec`][docarray.array.doc_vec.doc_vec.DocVec] to represent multi-modal data and send it over the wire.
In this section we will see how to store and persist this data.

DocArray offers to ways of storing your data:

1. In a **[Document Store](#document-store)** for simple long-term storage
2. In a **[Document Index](#document-index)** for fast retrieval using vector similarity

## Document Store
    
[DocList][docarray.array.doc_list.doc_list.DocList] can be persisted using the
[`.push()`][docarray.array.doc_list.pushpull.PushPullMixin.push] and 
[`.pull()`][docarray.array.doc_list.pushpull.PushPullMixin.pull] methods. 
Under the hood, [DocStore][docarray.store.abstract_doc_store.AbstractDocStore] is used to persist a `DocList`. 
You can store your documents on-disk. Alternatively, you can upload them to [AWS S3](https://aws.amazon.com/s3/), 
[minio](https://min.io) or [Jina AI Cloud](https://cloud.jina.ai/user/storage). 

This section covers the following three topics:

  - [Store](doc_store/store_file.md) of [`BaseDoc`][docarray.base_doc.doc.BaseDoc], [`DocList`][docarray.array.doc_list.doc_list.DocList] and [`DocVec`][docarray.array.doc_vec.doc_vec.DocVec] on-disk
  - [Store on Jina AI Cloud](doc_store/store_jac.md) 
  - [Store on S3](doc_store/store_s3.md)
   
## Document Index
