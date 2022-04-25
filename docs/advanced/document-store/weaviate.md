(weaviate)=
# Weaviate

One can use [Weaviate](https://www.semi.technology) as the document store for DocumentArray. It is useful when one wants to have faster Document retrieval on embeddings, i.e. `.match()`, `.find()`.

````{tip}
This feature requires `weaviate-client`. You can install it via `pip install "docarray[full]".` 
````

Here is a video tutorial that guides you to build a simple image search using Weaviate and Docarray.

<center>
<iframe width="560" height="315" src="https://www.youtube.com/embed/rBKvoIGihnY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</center>

## Usage

### Start Weaviate service

To use Weaviate as the storage backend, it is required to have the Weaviate service started. Create `docker-compose.yml` as follows:

```yaml
---
version: '3.4'
services:
  weaviate:
    command:
      - --host
      - 0.0.0.0
      - --port
      - '8080'
      - --scheme
      - http
    image: semitechnologies/weaviate:1.11.0
    ports:
      - "8080:8080"
    restart: on-failure:0
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'none'
      ENABLE_MODULES: ''
      CLUSTER_HOSTNAME: 'node1'
...
```

Then

```bash
docker compose up
```

### Create DocumentArray with Weaviate backend

Assuming service is started using the default configuration (i.e. server address is `http://localhost:8080`), one can instantiate a DocumentArray with Weaviate storage as such:

```python
from docarray import DocumentArray

da = DocumentArray(storage='weaviate')
```

The usage would be the same as the ordinary DocumentArray.

To access a DocumentArray formerly persisted, one can specify the name, the host, the port and the protocol to connect to the server. `name` is required in this case but other connection parameters are optional. If they are not provided, then it will connect to the Weaviate service bound to `http://localhost:8080`.

Note, that the `name` parameter in `config` needs to be capitalized.

```python
from docarray import DocumentArray

da = DocumentArray(
    storage='weaviate', config={'name': 'Persisted', 'host': 'localhost', 'port': 1234}
)

da.summary()
```

Other functions behave the same as in-memory DocumentArray.

## Config

The following configs can be set:

| Name               | Description                                                                                                                                                    | Default                                            |
|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------|
| `host`             | Hostname of the Weaviate server                                                                                                                                | 'localhost'                                        |
| `port`             | port of the Weaviate server                                                                                                                                    | 8080                                               |
| `protocol`         | protocol to be used. Can be 'http' or 'https'                                                                                                                  | 'http'                                             |
| `name`             | Weaviate class name; the class name of Weaviate object to presesent this DocumentArray                                                                         | None                                               |
| `serialize_config` | [Serialization config of each Document](../../fundamentals/document/serialization.md)                                                                          | None                                               |
| `ef`               | The size of the dynamic list for the nearest neighbors (used during the search). The higher ef is chosen, the more accurate, but also slower a search becomes. | `None`, defaults to the default value in Weaviate* |
| `ef_construction`  | The size of the dynamic list for the nearest neighbors (used during the construction). Controls index search speed/build speed tradeoff.                       | `None`, defaults to the default value in Weaviate* |
| `timeout_config` |  Set the timeout configuration for all requests to the Weaviate server.                                                                                          | `None`, defaults to the default value in Weaviate* |
| `max_connections`  | The maximum number of connections per element in all layers.                                                                                                   | `None`, defaults to the default value in Weaviate* |

*You can read more about the HNSW parameters and their default values [here](https://weaviate.io/developers/weaviate/current/vector-index-plugins/hnsw.html#how-to-use-hnsw-and-parameters)

## Minimum Example

The following example shows how to use DocArray with Weaviate Document Store in order to index and search text 
Documents.

First, let's run the create the `DocumentArray` instance (make sure a Weaviate server is up and running):

```python
from docarray import DocumentArray

da = DocumentArray(
    storage="weaviate", config={"name": "Persisted", "host": "localhost", "port": 8080}
)
```

Then, we can index some Documents:

```python
from docarray import Document

da.extend(
    [
        Document(text='Persist Documents with Weaviate.'),
        Document(text='And enjoy fast nearest neighbor search.'),
        Document(text='All while using DocArray API.'),
    ]
)
```

Now, we can generate embeddings inside the database using BERT model:

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')


def collate_fn(da):
    return tokenizer(da.texts, return_tensors='pt', truncation=True, padding=True)


da.embed(model, collate_fn=collate_fn)
```


Finally, we can query the database and print the results:

```python
results = da.find(
    DocumentArray([Document(text='How to persist Documents')]).embed(
        model, collate_fn=collate_fn
    ),
    limit=1,
)

print(results[0].text)
```

```text
Persist Documents with Weaviate.
```
