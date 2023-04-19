# Store on-disk

When you want to use your [DocList][docarray.array.doc_list.doc_list.DocList] in another place, you can use:

- the [`.push()`][docarray.array.doc_list.pushpull.PushPullMixin.push] method to push the [DocList][docarray.array.doc_list.doc_list.DocList] 
to one place.
- the [`.pull()`][docarray.array.doc_list.pushpull.PushPullMixin.pull] method to pull its content back. 

## Push and pull

To use the store locally, you need to pass a local file path to the function starting with `'file://'`.

```python
from docarray import BaseDoc, DocList


class SimpleDoc(BaseDoc):
    text: str


dl = DocList[SimpleDoc]([SimpleDoc(text=f'doc {i}') for i in range(8)])
dl.push('file://simple_dl')

dl_pull = DocList[SimpleDoc].pull('file://simple_dl')
```

A file with the name of `simple_dl.docs` will be created in `$HOME/.docarray/cache` to store the `DocList`.


## Push and pull with streaming

When you have a large amount of documents to push and pull, you can use the streaming method:
[`.push_stream()`][docarray.array.doc_list.pushpull.PushPullMixin.push_stream] and 
[`.pull_stream()`][docarray.array.doc_list.pushpull.PushPullMixin.pull_stream] stream the `DocList`
to save memory usage. You set multiple `DocList`s to pull from the same source as well:

```python
from docarray import BaseDoc, DocList


class SimpleDoc(BaseDoc):
    text: str


store_docs = [SimpleDoc(text=f'doc {i}') for i in range(8)]

DocList[SimpleDoc].push_stream(
    iter(store_docs),
    'file://dl_stream',
)
dl_pull_stream_1 = DocList[SimpleDoc].pull_stream('file://dl_stream')
dl_pull_stream_2 = DocList[SimpleDoc].pull_stream('file://dl_stream')

for d1, d2 in zip(dl_pull_stream_1, dl_pull_stream_2):
    print(f'get {d1}, get {d2}')
```

<details>
    <summary>Output</summary>
    ```text
    get SimpleDoc(id='5a4b92af27aadbb852d636892506998b', text='doc 0'), get SimpleDoc(id='5a4b92af27aadbb852d636892506998b', text='doc 0')
    get SimpleDoc(id='705e4f6acbab0a6ff10d11a07c03b24c', text='doc 1'), get SimpleDoc(id='705e4f6acbab0a6ff10d11a07c03b24c', text='doc 1')
    get SimpleDoc(id='4fb5c01bd5f935bbe91cf73e271ad590', text='doc 2'), get SimpleDoc(id='4fb5c01bd5f935bbe91cf73e271ad590', text='doc 2')
    get SimpleDoc(id='381498cef78f1d4f1d80415d67918940', text='doc 3'), get SimpleDoc(id='381498cef78f1d4f1d80415d67918940', text='doc 3')
    get SimpleDoc(id='d968bc6fa235b1cfc69eded92926157e', text='doc 4'), get SimpleDoc(id='d968bc6fa235b1cfc69eded92926157e', text='doc 4')
    get SimpleDoc(id='30bf347427a4bd50ce8ada1841320fe3', text='doc 5'), get SimpleDoc(id='30bf347427a4bd50ce8ada1841320fe3', text='doc 5')
    get SimpleDoc(id='1389877ac97b3e6d0e8eb17568934708', text='doc 6'), get SimpleDoc(id='1389877ac97b3e6d0e8eb17568934708', text='doc 6')
    get SimpleDoc(id='264b0eff2cd138d296f15c685e15bf23', text='doc 7'), get SimpleDoc(id='264b0eff2cd138d296f15c685e15bf23', text='doc 7')
    ```
</details>
