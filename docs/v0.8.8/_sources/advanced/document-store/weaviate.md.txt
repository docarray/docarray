# Weaviate

One can use [Weaviate](https://www.semi.technology) as the document store for DocumentArray. It is useful when one wants to have faster Document retrieval on embeddings, i.e. `.match()`, `.find()`.


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
    image: semitechnologies/weaviate:1.10.0
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

da = DocumentArray(storage='weaviate', config={'name': 'Persisted', 'host': 'localhost', 'port': 1234})

da.summary()
```

Other functions behave the same as in-memory DocumentArray.

## Config

The following configs can be set:

| Name               | Description                                                                            | Default                     |
|--------------------|----------------------------------------------------------------------------------------|-----------------------------|
| `host`             | Hostname of the Weaviate server                                                        | 'localhost'                 |
| `port`             | port of the Weaviate server                                                            | 8080                        |
| `protocol`         | protocol to be used. Can be 'http' or 'https'                                          | 'http'                      |
| `name`             | Weaviate class name; the class name of Weaviate object to presesent this DocumentArray | None                        |
| `serialize_config` | [Serialization config of each Document](../../fundamentals/document/serialization.md)  | None                        |
