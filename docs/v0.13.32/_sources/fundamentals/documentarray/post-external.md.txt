(da-post)=
# Process via External Flow or Executor

```{tip}
This feature requires `jina` dependency. Please install Jina via `pip install -U jina`.
```

You can call an external Flow/Sandbox/Executor to "process" a DocumentArray via {meth}`~docarray.array.mixins.post.PostMixin.post`. The external Flow/Executor can be either local, remote, or inside Docker container.

For example, to use an existing Flow on `192.168.2.3` on port `12345` to process a DocumentArray:

```python
from docarray import DocumentArray

da = DocumentArray.empty(10)

r = da.post('grpc://192.168.2.3:12345')
r.summary()
```

One can also use any [Executor from Jina Hub](https://hub.jina.ai), e.g.
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

Single Document has a sugar syntax that leverages this feature. Hence the above example can be also written as follows:

```python
from docarray import Document

d = Document(text='Hi Alex, are you with your laptop?')
r = d.post('jinahub+sandbox://CoquiTTS7')
```

## Accept schemes

{meth}`~docarray.array.mixins.post.PostMixin.post` accepts a URI-like scheme that supports a wide range of Flow/Hub Executor. It is described as below:

```text
scheme://netloc[:port][/path]
```

| Attribute | Supported Values                                        | Meaning                                                                                                                              |
|-----------|---------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| `scheme`  | 1. One of `grpc`, `websocket`, `http`                   | `protocol` of the connected Flow                                                                                                     |
|           | 2. One of `jinahub`, `jinahub+docker`, `jinhub+sandbox` | Jina hub executor in source code, Docker container, sandbox                                                                          |
| `netloc`  | 1. Host address                                         | `host` of the connected Flow                                                                                                         |
|   | 2. Hub Executor name                                    | Any Executor [listed here](https://hub.jina.ai)                                                                                      |
|   | 3. Executor version(optional)                           | Such as v0.1.1, v0.1.1-gpu, by default latest                                                                                        |
| `:port` | e.g. `:55566`                                           | `port` of the connected Flow. This is required when using `scheme` type (1) ; it is ignored when using hub-related `scheme` type (2) |
| `/path` | e.g. `/foo`                                             | The endpoint of the Executor you want to call.                                                                                       |


Some examples:
- `.post('websocket://localhost:8081/foo')`: call the `/foo` endpoint of the Flow on `localhost` port `8081` with `websocket` protocol to process the DocumentArray; processing is on local.
- `.post('grpc://192.168.12.2:12345/foo')`: call the `/foo` endpoint of the Flow on `192.168.12.2` port `12345` with `grpc` protocol to process the DocumentArray; processing is on remote.
- `.post('jinahub://Hello/foo')`: call the `/foo` endpoint of the Hub Executor `Hello` to process the DocumentArray; porcessing is on local.
- `.post('jinahub+sandbox://Hello/foo')`: call the `/foo` endpoint of the Hub Sandbox `Hello` to process the DocumentArray; porcessing is on remote.
- `.post('jinahub+docker://Hello/v0.5.0/foo')`: call the `/foo` endpoint of the Hub Sandbox `Hello` of version `v0.5.0` to process the DocumentArray; porcessing in container.

## Read more

For more explanation of Flow, Hub Executor and Sandbox, please refer to [Jina docs](https://docs.jina.ai).
