# Elasticsearch

One can use [Elasticsearch](https://www.elastic.co) as the document store for DocumentArray. It is useful when one wants to have faster Document retrieval on embeddings, i.e. `.match()`, `.find()`.

````{tip}
This feature requires `elasticsearch`. You can install it via `pip install "docarray[full]".` 
````

## Usage

### Start Elastic service

To use Elasticsearch as the storage backend, it is required to have the Elasticsearch service started. Create `docker-compose.yml` as follows:

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


### Create DocumentArray with Elasticsearch backend

Assuming service is started using the default configuration (i.e. server address is `http://localhost:9200`), one can instantiate a DocumentArray with Elasticsearch storage as such:

```python
from docarray import DocumentArray

da = DocumentArray(storage='elasticsearch', config={'n_dim': 128})
```

The usage would be the same as the ordinary DocumentArray, but the dimension of an embedding for a Document must be provided at creation time.

### Secure connection

By default, Elasticsearch server runs with security layer that disables the plain HTTP connection. You can pass the `host` with `api_id` or `ca_certs` inside `es_config` to the constructor. For example,

```python
from docarray import DocumentArray

da = DocumentArray(
    storage='elasticsearch',
    config={
        'hosts': 'https://elastic:PRq7je_hJ4i4auh+Hq+*@localhost:9200',
        'n_dim': 128,
        'es_config': {'ca_certs': '/Users/hanxiao/http_ca.crt'},
    },
)
```

Here is [the official Documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html#elasticsearch-security-certificates) for you to get certificate, password etc.

To access a DocumentArray formerly persisted, one can specify `index_name` and the hosts.

The following example will build a DocumentArray with previously stored data from `old_stuff` on `http://localhost:9200`:

```python
from docarray import DocumentArray, Document

da = DocumentArray(
    storage='elasticsearch',
    config={'index_name': 'old_stuff', 'n_dim': 128},
)

da.extend([Document() for _ in range(1000)])

da2 = DocumentArray(
    storage='elasticsearch',
    config={'index_name': 'old_stuff', 'n_dim': 128},
)

da2.summary()
```

```text
              Documents Summary               
                                              
  Length                 2000                 
  Homogenous Documents   True                 
  Common Attributes      ('id', 'embedding')  
                                              
                      Attributes Summary                       
                                                               
  Attribute   Data type      #Unique values   Has empty value  
 ───────────────────────────────────────────────────────────── 
  embedding   ('ndarray',)   1000             False            
  id          ('str',)       1000             False            
                                                               
              Storage Summary               
                                            
  Backend            ElasticSearch          
  Host               http://localhost:9200  
  Distance           cosine                 
  Vector dimension   128                    
  ES config          {}                     
                                            
[0.14890289 0.3168339  0.03050802 0.06785086 0.94719299 0.32490566
 ...]
```

Other functions behave the same as in-memory DocumentArray.

## Config

The following configs can be set:

| Name         | Description                                                                                      | Default     |
|--------------|--------------------------------------------------------------------------------------------------|-------------|
| `hosts`      | Hostname of the Elasticsearch server                                                             | `http://localhost:9200` |
| `es_config`  | Other ES configs in a Dict and pass to `Elasticsearch` client constructor, e.g. `cloud_id`, `api_key`| None |
| `index_name` | Elasticsearch index name; the class name of Elasticsearch index object to set this DocumentArray | None        |
| `n_dim`      | Dimensionality of the embeddings                                                                 | None        |
| `distance`   | Similarity metric in Elasticsearch                                                               | `cosine`|


```{tip}
Note that it is plural `hosts` not `host`, to comply with Elasticsearch client's interface.
```
