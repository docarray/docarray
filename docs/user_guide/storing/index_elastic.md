# Elasticsearch Document Index

DocArray comes with two Document Indexes for [Elasticsearch](https://www.elastic.co/elasticsearch/):

- [ElasticDocIndex][docarray.index.backends.elastic.ElasticDocIndex], based on [Elasticsearch 8](https://github.com/elastic/elasticsearch).
- [ElasticV7DocIndex][docarray.index.backends.elasticv7.ElasticV7DocIndex], based on [Elasticsearch 7.10](https://www.elastic.co/downloads/past-releases/elasticsearch-7-10-0).

!!! tip "Should you use ES v7 or v8?"
    [Elasticsearch v8](https://www.elastic.co/blog/whats-new-elastic-8-0-0) is the current version of ES and offers
    **native vector search (ANN) support**, alongside text and range search.

    [Elasticsearch v7.10](https://www.elastic.co/downloads/past-releases/elasticsearch-7-10-0) can store vectors, but
    **does _not_ support native ANN vector search**, but only exhaustive (i.e. slow) vector search, alongside text and range search.

    Some users prefer to use ES v7.10 because it is available under a [different license](https://www.elastic.co/pricing/faq/licensing) to ES v8.0.0.

!!! note "Installation"
    To use [ElasticDocIndex][docarray.index.backends.elastic.ElasticDocIndex], you need to install the following dependencies:

    ```console
    pip install elasticsearch==8.6.2
    pip install elastic-transport
    ```

    To use [ElasticV7DocIndex][docarray.index.backends.elasticv7.ElasticV7DocIndex], you need to install the following dependencies:
    
    ```console
    pip install elasticsearch==7.10.1
    pip install elastic-transport
    ```


The following example is based on [ElasticDocIndex][docarray.index.backends.elastic.ElasticDocIndex],
but will also work for [ElasticV7DocIndex][docarray.index.backends.elasticv7.ElasticV7DocIndex].

# Start Elasticsearch

You can use docker-compose to create a local Elasticsearch service with the following `docker-compose.yml`.

```yaml
version: "3.3"
services:
  elastic:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.6.2
    environment:
      - xpack.security.enabled=false
      - discovery.type=single-node
      - ES_JAVA_OPTS=-Xmx1024m
    ports:
      - "9200:9200"
    networks:
      - elastic

networks:
  elastic:
    name: elastic
```

Run the following command in the folder of the above `docker-compose.yml` to start the service:

```bash
docker-compose up
```

## Construct

To construct an index, you first need to define a schema in the form of a `Document`.

There are a number of configurations you can pack into your schema:

- Every field in your schema will become one column in the database
- For vector fields, such as `NdArray`, `TorchTensor`, or `TensorflowTensor`, you need to specify a dimensionality to be able to perform vector search
- You can override the default column type for every field by passing any [ES field data type](https://www.elastic.co/guide/en/elasticsearch/reference/8.6/mapping-types.html) to `field_name: Type = Field(col_type=...)`. You can see an example of this in the [section on keyword filters](#keyword-filter).

Additionally, you can pass a `hosts` argument to the `__init__()` method to connect to an ES instance.
By default, it is `http://localhost:9200`. 

```python
import numpy as np
from pydantic import Field

from docarray import BaseDoc
from docarray.index import ElasticDocIndex
from docarray.typing import NdArray


class SimpleDoc(BaseDoc):
    # specify tensor field with dimensionality 128
    tensor: NdArray[128]
    # alternative and equivalent definition:
    # tensor: NdArray = Field(dims=128)


doc_index = ElasticDocIndex[SimpleDoc](hosts='http://localhost:9200')
```

## Index documents

Use `.index()` to add documents into the index.
The`.num_docs()` method returns the total number of documents in the index.

```python
index_docs = [SimpleDoc(tensor=np.ones(128)) for _ in range(64)]

doc_index.index(index_docs)

print(f'number of docs in the index: {doc_index.num_docs()}')
```

## Access documents

To access the `Doc`, you need to specify the `id`. You can also pass a list of `id` to access multiple documents.

```python
# access a single Doc
doc_index[index_docs[16].id]

# access multiple Docs
doc_index[index_docs[16].id, index_docs[17].id]
```

### Persistence

You can hook into a database index that was persisted during a previous session.
To do so, you need to specify `index_name` and the `hosts`:

```python
doc_index = ElasticDocIndex[SimpleDoc](
    hosts='http://localhost:9200', index_name='previously_stored'
)
doc_index.index(index_docs)

doc_index2 = ElasticDocIndex[SimpleDoc](
    hosts='http://localhost:9200', index_name='previously_stored'
)

print(f'number of docs in the persisted index: {doc_index2.num_docs()}')
```


## Delete documents

To delete the documents, use the built-in function `del` with the `id` of the Documents that you want to delete.
You can also pass a list of `id`s to delete multiple documents.

```python
# delete a single Doc
del doc_index[index_docs[16].id]

# delete multiple Docs
del doc_index[index_docs[17].id, index_docs[18].id]
```

## Find nearest neighbors

The `.find()` method is used to find the nearest neighbors of a vector.

You need to specify the `search_field` that is used when performing the vector search.
This is the field that serves as the basis of comparison between your query and indexed Documents.

You can use the `limit` argument to configure how many documents to return.

!!! note
    [ElasticV7DocIndex][docarray.index.backends.elasticv7.ElasticV7DocIndex] uses Elasticsearch v7.10.1, which does not support approximate nearest neighbour algorithms such as HNSW.
    This can lead to poor performance when the search involves many vectors.
    [ElasticDocIndex][docarray.index.backends.elastic.ElasticDocIndex] does not have this limitation.

```python
query = SimpleDoc(tensor=np.ones(128))

docs, scores = doc_index.find(query, limit=5, search_field='tensor')
```

## Nested data

When using the index you can define multiple fields, including nesting documents inside another document.

Consider the following example:

- You have `YouTubeVideoDoc` including the `tensor` field calculated based on the description.
- `YouTubeVideoDoc` has `thumbnail` and `video` fields, each with their own `tensor`.

```python
from docarray.typing import ImageUrl, VideoUrl, AnyTensor


class ImageDoc(BaseDoc):
    url: ImageUrl
    tensor: AnyTensor = Field(similarity='cosine', dims=64)


class VideoDoc(BaseDoc):
    url: VideoUrl
    tensor: AnyTensor = Field(similarity='cosine', dims=128)


class YouTubeVideoDoc(BaseDoc):
    title: str
    description: str
    thumbnail: ImageDoc
    video: VideoDoc
    tensor: AnyTensor = Field(similarity='cosine', dims=256)


doc_index = ElasticDocIndex[YouTubeVideoDoc]()
index_docs = [
    YouTubeVideoDoc(
        title=f'video {i+1}',
        description=f'this is video from author {10*i}',
        thumbnail=ImageDoc(url=f'http://example.ai/images/{i}', tensor=np.ones(64)),
        video=VideoDoc(url=f'http://example.ai/videos/{i}', tensor=np.ones(128)),
        tensor=np.ones(256),
    )
    for i in range(8)
]
doc_index.index(index_docs)
```

**You can perform search on any nesting level** by using the dunder operator to specify the field defined in the nested data.

In the following example, you can see how to perform vector search on the `tensor` field of the `YouTubeVideoDoc` or the `tensor` field of the `thumbnail` and `video` field:

```python
# example of find nested and flat index
query_doc = YouTubeVideoDoc(
    title=f'video query',
    description=f'this is a query video',
    thumbnail=ImageDoc(url=f'http://example.ai/images/1024', tensor=np.ones(64)),
    video=VideoDoc(url=f'http://example.ai/videos/1024', tensor=np.ones(128)),
    tensor=np.ones(256),
)

# find by the youtubevideo tensor
docs, scores = doc_index.find(query_doc, search_field='tensor', limit=3)

# find by the thumbnail tensor
docs, scores = doc_index.find(query_doc, search_field='thumbnail__tensor', limit=3)

# find by the video tensor
docs, scores = doc_index.find(query_doc, search_field='video__tensor', limit=3)
```

To delete a nested data, you need to specify the `id`.

!!! note
    You can only delete `Doc` at the top level. Deletion of `Doc`s on lower levels is not yet supported.

```python
# example of delete nested and flat index
del doc_index[index_docs[3].id, index_docs[4].id]
```

## Other Elasticsearch queries

Besides vector search, you can also perform other queries supported by Elasticsearch, such as text search, and various filters.

### Text search

As in "pure" Elasticsearch, you can use text search directly on the field of type `str`:

```python
class NewsDoc(BaseDoc):
    text: str


doc_index = ElasticDocIndex[NewsDoc]()
index_docs = [
    NewsDoc(id='0', text='this is a news for sport'),
    NewsDoc(id='1', text='this is a news for finance'),
    NewsDoc(id='2', text='this is another news for sport'),
]
doc_index.index(index_docs)
query = 'finance'

# search with text
docs, scores = doc_index.text_search(query, search_field='text')
```

### Query Filter

The `filter()` method accepts queries that follow the [Elasticsearch Query DSL](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl.html) and consists of leaf and compound clauses.

Using this, you can perform [keyword filters](#keyword-filter), [geolocation filters](#geolocation-filter) and [range filters](#range-filter).

#### Keyword filter

To filter documents in your index by keyword, you can use `Field(col_type='keyword')` to enable keyword search for given fields:

```python
class NewsDoc(BaseDoc):
    text: str
    category: str = Field(col_type='keyword')  # enable keyword filtering


doc_index = ElasticDocIndex[NewsDoc]()
index_docs = [
    NewsDoc(id='0', text='this is a news for sport', category='sport'),
    NewsDoc(id='1', text='this is a news for finance', category='finance'),
    NewsDoc(id='2', text='this is another news for sport', category='sport'),
]
doc_index.index(index_docs)

# search with filer
query_filter = {'terms': {'category': ['sport']}}
docs = doc_index.filter(query_filter)
```

#### Geolocation filter

To filter documents in your index by geolocation, you can use `Field(col_type='geo_point')` on a given field:

```python
class NewsDoc(BaseDoc):
    text: str
    location: dict = Field(col_type='geo_point')  # enable geolocation filtering


doc_index = ElasticDocIndex[NewsDoc]()
index_docs = [
    NewsDoc(text='this is from Berlin', location={'lon': 13.24, 'lat': 50.31}),
    NewsDoc(text='this is from Beijing', location={'lon': 116.22, 'lat': 39.55}),
    NewsDoc(text='this is from San Jose', location={'lon': -121.89, 'lat': 37.34}),
]
doc_index.index(index_docs)

# filter the eastern hemisphere
query = {
    'bool': {
        'filter': {
            'geo_bounding_box': {
                'location': {
                    'top_left': {'lon': 0, 'lat': 90},
                    'bottom_right': {'lon': 180, 'lat': 0},
                }
            }
        }
    }
}

docs = doc_index.filter(query)
```

#### Range filter

You can have [range field types](https://www.elastic.co/guide/en/elasticsearch/reference/8.6/range.html) in your document schema and set `Field(col_type='integer_range')`(or also `date_range`, etc.) to filter documents based on the range of the field. 

```python
class NewsDoc(BaseDoc):
    time_frame: dict = Field(
        col_type='date_range', format='yyyy-MM-dd'
    )  # enable range filtering


doc_index = ElasticDocIndex[NewsDoc]()
index_docs = [
    NewsDoc(time_frame={'gte': '2023-01-01', 'lt': '2023-02-01'}),
    NewsDoc(time_frame={'gte': '2023-02-01', 'lt': '2023-03-01'}),
    NewsDoc(time_frame={'gte': '2023-03-01', 'lt': '2023-04-01'}),
]
doc_index.index(index_docs)

query = {
    'bool': {
        'filter': {
            'range': {
                'time_frame': {
                    'gte': '2023-02-05',
                    'lt': '2023-02-10',
                    'relation': 'contains',
                }
            }
        }
    }
}

docs = doc_index.filter(query)
```

### Hybrid serach and query builder

To combine any of the "atomic" search approaches above, you can use the `QueryBuilder` to build your own hybrid query. 

For this the `find()`, `filter()` and `text_search()` methods and their combination are supported.

For example, you can build a hybrid serach query that performs range filtering, vector search and text search:

```python
class MyDoc(BaseDoc):
    tens: NdArray[10] = Field(similarity='l2_norm')
    num: int
    text: str


doc_index = ElasticDocIndex[MyDoc]()
index_docs = [
    MyDoc(id=f'{i}', tens=np.ones(10) * i, num=int(i / 2), text=f'text {int(i/2)}')
    for i in range(10)
]
doc_index.index(index_docs)

q = (
    doc_index.build_query()
    .filter({'range': {'num': {'lte': 3}}})
    .find(index_docs[-1], search_field='tens')
    .text_search('0', search_field='text')
    .build()
)
docs, _ = doc_index.execute_query(q)
```

You can also manually build a valid ES query and directly pass it to the `execute_query()` method.

## Configuration options

### DBConfig

The following configs can be set in `DBConfig`:

| Name              | Description                                                                                                                            | Default                 |
|-------------------|----------------------------------------------------------------------------------------------------------------------------------------|-------------------------|
| `hosts`           | Hostname of the Elasticsearch server                                                                                                   | `http://localhost:9200` |
| `es_config`       | Other ES [configuration options](https://www.elastic.co/guide/en/elasticsearch/client/python-api/8.6/config.html) in a Dict and pass to `Elasticsearch` client constructor, e.g. `cloud_id`, `api_key` | None |
| `index_name`      | Elasticsearch index name, the name of Elasticsearch index object                                       | None. Data will be stored in an index named after the Document type used as schema. |
| `index_settings`  | Other [index settings](https://www.elastic.co/guide/en/elasticsearch/reference/8.6/index-modules.html#index-modules-settings) in a Dict for creating the index    | dict  |
| `index_mappings`  | Other [index mappings](https://www.elastic.co/guide/en/elasticsearch/reference/8.6/mapping.html) in a Dict for creating the index | dict  |

You can pass any of the above as keyword arguments to the `__init__()` method or pass an entire configuration object.
See [here](docindex.md#configuration-options#customize-configurations) for more information.

### RuntimeConfig

The `RuntimeConfig` dataclass of `ElasticDocIndex` consists of `default_column_config` and `chunk_size`. You can change `chunk_size` for batch operations:

```python
doc_index = ElasticDocIndex[SimpleDoc]()
doc_index.configure(ElasticDocIndex.RuntimeConfig(chunk_size=1000))
```

`default_column_config` is the default configurations for every column type. Since there are many column types in Elasticsearch, you can also consider changing the column config when defining the schema.

```python
class SimpleDoc(BaseDoc):
    tensor: NdArray[128] = Field(similarity='l2_norm', m=32, num_candidates=5000)


doc_index = ElasticDocIndex[SimpleDoc]()
```

You can pass the above as keyword arguments to the `configure()` method or pass an entire configuration object.
See [here](docindex.md#configuration-options#customize-configurations) for more information.
