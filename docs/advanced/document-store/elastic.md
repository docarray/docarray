# Elastic

One can use [Elastic](https://www.elastic.co) as the document store for DocumentArray. It is useful when one wants to have faster Document retrieval on embeddings, i.e. `.match()`, `.find()`.

````{tip}
This feature requires `elasticsearch-client`. You can install it via `pip install "docarray[full]".` 
````

## Usage

### Start Elastic service

To use Elastic as the storage backend, it is required to have the Elastic service started. Create `docker-compose.yml` as follows:

```yaml
version: "3.3"
services:
  elastic:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.1.0
    environment:
      - xpack.security.enabled=false
      - discovery.type=single-node
    ports:
      - "9200:9200"
    networks:
      - elastic

networks:
  elastic:
    name: elastic
```

Then

```bash
docker-compose up
```

### Create DocumentArray with Elastic backend

Assuming service is started using the default configuration (i.e. server address is `http://localhost:9200`), one can instantiate a DocumentArray with Elastic storage as such:

```python
from docarray import DocumentArray

da = DocumentArray(storage='elastic',config={'n_dim':128})
```

The usage would be the same as the ordinary DocumentArray, but the dimension of an embedding for a Document must be provided at creation time.

To access a DocumentArray formerly persisted, one can specify the index name, the host, the port and the protocol to connect to the server. If they are not provided, then it will connect to the Elastic service bound to `http://localhost:9200`.

```python
from docarray import DocumentArray

da = DocumentArray(storage='elastic', config={'index_name':'ndim_128', 'n_dim':128, 'port': 9200})
```

Other functions behave the same as in-memory DocumentArray.

## Config

The following configs can be set:

| Name               | Description                                                                           | Default     |
|--------------------|---------------------------------------------------------------------------------------|-------------|
| `host`             | Hostname of the Elastic server                                                        | 'localhost' |
| `port`             | port of the Elastic server                                                            | 9200        |
| `protocol`         | protocol to be used. Can be 'http' or 'https'                                         | 'http'      |
| `index_name`       | Elastic index name; the class name of Elastic index object to set this DocumentArray  | None        |

