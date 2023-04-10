# Store
[DocList][docarray.array.doc_list.doc_list.DocList] can be persisted using `push()` and `pull()` functions. Under the hood, 
[DocStore][docarray.store.abstract_doc_store.AbstractDocStore] is used to persist a `DocList`. You can store your `Doc` on-disk. Alternatively, you can upload to [AWS S3](https://aws.amazon.com/s3/), [minio](https://min.io) or [Jina AI Cloud](https://cloud.jina.ai/user/storage). 

# Store on-disk
When you want to use your `DocList` in another place, you can use the `push()` function to push the `DocList` to one place and later use the `pull()` function to pull its content back. 

## Push & pull
To use the store locally, you need to pass a local file path to the function starting with `file://`.

```python
from docarray import BaseDoc, DocList


class SimpleDoc(BaseDoc):
    text: str


store_docs = [SimpleDoc(text=f'doc {i}') for i in range(8)]

dl = DocList[SimpleDoc]()
dl.extend([SimpleDoc(text=f'doc {i}') for i in range(8)])
dl.push('file:///Users/docarray/tmp/simple_dl')

dl_pull = DocList[SimpleDoc].pull('file:///Users/docarray/tmp/simple_dl')
```

Under `/Users/docarray/tmp/`, there is a file with the name of `simple_dl.docs` being created to store the `DocList`.
```output
tmp
└── simple_dl.docs
```

## Push & Pull with streaming
When you have a large amount of `Doc` to push and pull, you could use the streaming function. `push_stream()` and `pull_stream()` can help you to stream the `DocList` in order to save the memory usage. You set multiple `DocList` to pull from the same source as well.

```python
from docarray import BaseDoc, DocList


class SimpleDoc(BaseDoc):
    text: str


store_docs = [SimpleDoc(text=f'doc {i}') for i in range(8)]

DocList[SimpleDoc].push_stream(iter(store_docs), 'file:///Users/docarray/tmp/dl_stream')
dl_pull_stream_1 = DocList[SimpleDoc].pull_stream('file:///Users/docarray/tmp/dl_stream')
dl_pull_stream_2 = DocList[SimpleDoc].pull_stream('file:///Users/docarray/tmp/dl_stream')
for d1, d2 in zip(dl_pull_stream_1, dl_pull_stream_2):
    print(f'get {d1}, get {d2}')
```

