# Qdrant

One can use [Qdrant](https://qdrant.tech) as the document store for DocumentArray. It is useful when one wants to have faster Document retrieval on embeddings, i.e. `.match()`, `.find()`.


## Usage

### Start Qdrant service

To use Qdrant as the storage backend, you need a running Qdrant server. You can use the Qdrant Docker image to run a 
server. Create `docker-compose.yml` as follows:

```yaml
---
version: '3.4'
services:
  qdrant:
    image: qdrant/qdrant:v0.5.1
    ports:
      - 6333:6333
    ulimits: # Only required for tests, as there are a lot of collections created
      nofile:
        soft: 65535
        hard: 65535
...
```

Then

```bash
docker compose up
```

### Create DocumentArray with Qdrant backend

Assuming service is started using the default configuration (i.e. server address is `http://localhost:6333`), one can 
instantiate a DocumentArray with Qdrant storage like so:

```python
from docarray import DocumentArray

da = DocumentArray(storage='qdrant', config={'n_dim': 10})
```

The usage would be the same as the ordinary DocumentArray.

To access a DocumentArray formerly persisted, one can specify the `collection_name`, the `host`  and the `port`. 


```python
from docarray import DocumentArray

da = DocumentArray(storage='qdrant', config={'collection_name': 'persisted', 'host': 'localhost', 'port': '6333', 'n_dim': 10})

da.summary()
```

Note that specifying the `n_dim` is mandatory before using Qdrant as a backend for DocumentArray.

Other functions behave the same as in-memory DocumentArray.

## Config

The following configs can be set:

| Name                 | Description                                                                     | Default                              |
|----------------------|---------------------------------------------------------------------------------|--------------------------------------|
| `n_dim`              | Number of dimensions of embeddings to be stored and retrieved                   | **This is always required**          |
| `collection_name`    | Qdrant collection name client                                                   | **Random collection name generated** |
| `host`               | Hostname of the Qdrant server                                                   | 'localhost'                          |
| `port`               | [port of the Qdrant server                                                      | 6333                                 |
| `distance`           | Distance metric to be used during search. Can be 'cosine', 'dot' or 'euclidean' | 'cosine'                             |
| `scroll_batch_size`  | batch size used when scrolling over the storage                                 | 64                                   |
