(da-post)=
# Process via External Flow or Executor

```{tip}
This feature requires the `jina` dependency. Please install it by running `pip install -U jina`.
```

You can call an external Flow/Sandbox/Executor to "process" a DocumentArray via {meth}`~docarray.array.mixins.post.PostMixin.post`. The external Flow/Executor can be local, remote, or inside a Docker container.

For example, to process a DocumentArray with an existing Flow at `192.168.2.3` on port `12345`:

```python
from docarray import DocumentArray

da = DocumentArray.empty(10)

r = da.post('grpc://192.168.2.3:12345')
r.summary()
```

You can also use any Executor from [Executor Hub](https://cloud.jina.ai):

```python
from docarray import DocumentArray, Document

da = DocumentArray([Document(text='Hi Alex, are you with your laptop?')])
r = da.post('jinahub+sandbox://CoquiTTS7', show_progress=True)

r.summary()
```

```text
                      Documents Summary

  Length                 1
  Homogenous Documents   True
  Common Attributes      ('id', 'mime_type', 'text', 'uri')

                     Attributes Summary

  Attribute   Data type   #Unique values   Has empty value
 ──────────────────────────────────────────────────────────
  id          ('str',)    1                False
  mime_type   ('str',)    1                False
  text        ('str',)    1                False
  uri         ('str',)    1                False
```

Single Documents have syntactic sugar that leverages this processing, meaning you can also write the above example as follows:

```python
from docarray import Document

d = Document(text='Hi Alex, are you with your laptop?')
r = d.post('jinahub+sandbox://CoquiTTS7')
```

## Accept schemes

{meth}`~docarray.array.mixins.post.PostMixin.post` accepts a URI-like scheme, supporting a wide range of Flows/Hub Executors:

```text
scheme://netloc[:port][/path]
```

| Attribute | Supported Values                                        | Meaning                                                                                                                              |
|-----------|---------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| `scheme`  | 1. One of `grpc`, `websocket`, `http`                   | `protocol` of connected Flow                                                                                                     |
|           | 2. One of `jinaai`, `jinaai+docker`, `jinaai+sandbox` | Executor Hub Executor in source code/Docker container/sandbox                                                                          |
| `netloc`  | 1. Host address                                         | `host` of connected Flow                                                                                                         |
|   | 2. Hub Executor name                                    | Any [Hub Executor](https://cloud.jina.ai)                                                                                      |
|   | 3. Executor version (optional)                           | e.g. `v0.1.1`, `v0.1.1-gpu`. `latest` by default                                                                                         |
| `:port` | e.g. `:55566`                                           | `port` of connected Flow. Required when using `scheme` type (1); ignored when using Hub-related `scheme` type (2) |
| `/path` | e.g. `/foo`                                             | Endpoint of Executor you want to call.                                                                                       |


Some examples:

- `.post('websocket://localhost:8081/foo')`: call the `/foo` endpoint of the Flow at `localhost` port `8081` with `websocket` protocol to process the DocumentArray; processing is local.
- `.post('grpc://192.168.12.2:12345/foo')`: call the `/foo` endpoint of the Flow at `192.168.12.2` port `12345` with `grpc` protocol to process the DocumentArray; processing is remote.
- `.post('jinahub://Hello/foo')`: call the `/foo` endpoint of the Hub Executor `Hello` to process the DocumentArray; processing is local.
- `.post('jinahub+sandbox://Hello/foo')`: call the `/foo` endpoint of the Hub Sandbox `Hello` to process the DocumentArray; processing is remote.
- `.post('jinahub+docker://Hello/v0.5.0/foo')`: call the `/foo` endpoint of the Hub Sandbox `Hello` of version `v0.5.0` to process the DocumentArray; processing in container.

## Read more

For a deeper explanation of Flow, Hub Executor and Sandbox, refer to [Jina's docs](https://docs.jina.ai).
