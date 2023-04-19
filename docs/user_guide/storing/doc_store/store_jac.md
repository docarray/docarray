# Store on Jina AI Cloud

When you want to use your [`DocList`][docarray.DocList] in another place, you can use:
- the [`.push()`][docarray.array.doc_list.pushpull.PushPullMixin.push] method to push the `DocList` to Jina AI Cloud .
- the [`.pull()`][docarray.array.doc_list.pushpull.PushPullMixin.pull] function to pull its content back. 

!!! note
    To store documents on Jina AI Cloud, you need to install the extra dependency with the following line:

    ```cmd
    pip install "docarray[jac]"
    ```

## Push and pull

To use the store [`DocList`][docarray.DocList] on Jina AI Cloud, you need to pass a Jina AI Cloud path to the function starting with `'jac://'`.

Before getting started, create an account at [Jina AI Cloud](http://cloud.jina.ai/) and a [Personal Access Token (PAT)](https://cloud.jina.ai/settings/tokens).

```python
from docarray import BaseDoc, DocList
import os


class SimpleDoc(BaseDoc):
    text: str


os.environ['JINA_AUTH_TOKEN'] = 'YOUR_PAT'
DL_NAME = 'simple-dl'
dl = DocList[SimpleDoc]([SimpleDoc(text=f'doc {i}') for i in range(8)])

# push to Jina AI Cloud
dl.push(f'jac://{DL_NAME}')

# pull from Jina AI Cloud
dl_pull = DocList[SimpleDoc].pull(f'jac://{DL_NAME}')
```

!!! note
    When using `.push()` and `.pull()`, `DocList` calls the default `boto3` client. Be sure your default session is correctly set up.

## Push and pull with streaming

When you have a large amount of documents to push and pull, you can use the streaming function. 
[`.push_stream()`][docarray.array.doc_list.pushpull.PushPullMixin.push_stream] and 
[`.pull_stream()`][docarray.array.doc_list.pushpull.PushPullMixin.pull_stream] stream the 
[`DocList`][docarray.DocList] to save memory usage. 
You can set multiple `DocList` to pull from the same source as well. 
The usage is the same as streaming with local files. 
Please refer to [push and pull with streaming with local files](store_file.md#push-and-pull-with-streaming).

## Delete

To delete the store, you need to use the static method [`.delete()`][docarray.store.jac.JACDocStore.delete] of [`JACDocStore`][docarray.store.jac.JACDocStore] class:

```python
from docarray.store import JACDocStore

JACDocStore.delete(f'jac://{DL_NAME}')
```
