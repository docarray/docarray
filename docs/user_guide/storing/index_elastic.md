# Elastic
[ElasticV7DocIndex](docarray.index.backends.elastic.ElasticV7DocIndex) implement the index based on [Elasticsearch 7.10](https://github.com/elastic/elasticsearch). This is an implementation with vectors stored and supporting text/range search.

!!! note
    To use [ElasticV7DocIndex][docarray.index.backends.elastic.ElasticV7DocIndex], one need to install the extra dependency with the following command

    ```console
    pip install "docarray[elasticsearch]"
    ```

[ElasticDocIndex](docarray.index.backends.elastic.ElasticDocIndex) is based on [Elasticsearch 8](https://github.com/elastic/elasticsearch) and supports hnsw based vector search as well.

!!! note
    To use [ElasticDocIndex][docarray.index.backends.elastic.ElasticDocIndex], one need to install the extra dependency with the following command

    ```console
    pip install elasticsearch==8.6.2
    pip install elastic-transport
    ```


The following examples is based on `ElasticDocIndex`. We use docker-compose to create a local elasticsearch service with the following `docker-compose.yml`.

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

Run the following command in the folder of the above `docker-compose.yml` to start the service,

```bash
docker-compose up
```

## Construct
To construct an index, you need to define the schema first. You can define the schema in the same way as defining a `Doc`. Dimensionality is necessary for vector space, you need to specify the shape or define it by `dims`. TODO: add links to the detailed explaination.

`hosts` is the argument for setting the elasticsearch hosts. By default, it is `http://localhost:9200`. 


```python
from pydantic import Field

from docarray import BaseDoc
from docarray.index import ElasticDocIndex
from docarray.typing import NdArray


class SimpleDoc(BaseDoc):
    tensor: NdArray = Field(dims=128)
    # tensor: NdArray[128]


doc_index = ElasticDocIndex[SimpleDoc]()
```
TODO some common info: specifying col_type, custom_config, Union etc.

## Index
Use `.index()` to add `Doc` into the index. You could use the same class as the schema for defining the `Doc`. Alternatively, you need to define the `Doc` following the schema of the index. `.num_docs()` returns the total number of `Doc` in the index.

```python
index_docs = [SimpleDoc(tensor=np.ones(128)) for _ in range(64)]

doc_index.index(index_docs)

print(f'number of docs in the index: {doc_index.num_docs()}')
```

## Access
To access the `Doc`, you need to specify the `id`. You can also pass a list of `id` to access multiple `Doc`.

```python
# access a single Doc
doc_index[index_docs[16].id]

# access multiple Docs
doc_index[index_docs[16].id, index_docs[17].id]
```

### Persistence
To access a `Doc` formerly persisted, you can specify `index_name` and the `hosts`.

```python
doc_index = ElasticDocIndex[SimpleDoc](index_name='previously_stored')
doc_index.index(index_docs)

doc_index2 = ElasticDocIndex[SimpleDoc](index_name='previously_stored')

print(f'number of docs in the persisted index: {doc_index2.num_docs()}')
```


## Delete
To delete the `Doc`, use the built-in function `del` with the `id` of the `Doc` to be deleted. You can also pass a list of `id` to delete multiple `Doc`.

```python
# delete a single Doc
del doc_index[index_docs[16].id]

# delete multiple Docs
del doc_index[index_docs[16].id, index_docs[17].id]
```

## Find Nearest Neighbors
Use `.find()` to find the nearest neighbors of a tensor. You can use `limit` argument to configurate how much `Doc` to return, and `search_field` argument to configurate the name of the field to search on.

```python
query = SimpleDoc(tensor=np.ones(128))

docs, scores = doc_index.find(query, limit=5)
```

!!! note
    [ElasticV7DocIndex][docarray.index.backends.elastic.ElasticV7DocIndex] is using Elasticsearch v7.10.1 which does not support approximate nearest neighbour algorithms as Hnswlib. This could lead to a poor performance when the search involves too many vectors.

## Nested Index
When using the index, you can define multiple fields as well as the nested structure. In the following example, you have `YouTubeVideoDoc` including the `tensor` field calculated based on the description. Besides, `YouTbueVideoDoc` has `thumbnail` and `video` field, each of which has its own `tensor`.

```python
from docarray import BaseDoc
from docarray.typing import ImageUrl, VideoUrl, AnyTensor
from docarray.index import ElasticDocIndex
import numpy as np
from pydantic import Field


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

Use the `search_field` to specify which field to be used when performing the vector search. You can use the dunder operator to specify the field defined in the nested data. In the following codes, you can perform vector search on the `tensor` field of the `YouTubeVideoDoc` or on the `tensor` field of the `thumbnail` and `video` field.

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
You can only delete `Doc` at the top level. Deletion of the `Doc` on the lower level is not supported yet.

```python
# example of delete nested and flat index
del doc_index[index_docs[16].id, index_docs[32].id]
```

TODO field_name of nested level

## Elasticsearch Query
Besides the vector search, you can also perform other queries supported by Elasticsearch.

### Text Search
As in elasticsearch, you could use text search directly on the field of type `str`. 

```python
from pydantic import Field

from docarray import BaseDoc
from docarray.index import ElasticDocIndex


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
To filter the docs, you can use `col_type` to configurate the fields. `filter()` accepts queries that follow [Elasticsearch Query DSL](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl.html) and consists of leaf and compound clauses.

#### Keyword filter
To filter the docs, you can use `col_type='keyword'` to configurate the keyword search for the fields.

```python
from pydantic import Field

from docarray import BaseDoc
from docarray.index import ElasticDocIndex


class NewsDoc(BaseDoc):
    text: str
    category: str = Field(col_type='keyword')


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
To filter the docs, you can use `col_type='geo_point'` to configurate the keyword search for the fields.

```python
from pydantic import Field

from docarray import BaseDoc
from docarray.index import ElasticDocIndex


class NewsDoc(BaseDoc):
    text: str
    location: dict = Field(col_type='geo_point')


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
You can use `col_type='date_range'` is used to filter the docs based on the range of the date. 
TODO: find a use case.


### QueryBuilder


## Batched Operation


## Config

### DBConfig
The following configs can be set in `DBConfig`:

| Name              | Description                                                                                                                            | Default                 |
|-------------------|----------------------------------------------------------------------------------------------------------------------------------------|-------------------------|
| `hosts`           | Hostname of the Elasticsearch server                                                                                                   | `http://localhost:9200` |
| `es_config`       | Other ES [configuration options](https://www.elastic.co/guide/en/elasticsearch/client/python-api/8.6/config.html) in a Dict and pass to `Elasticsearch` client constructor, e.g. `cloud_id`, `api_key` | None |
| `index_name`      | Elasticsearch index name, the name of Elasticsearch index object                                       | None |
| `index_settings`  | Other [index settings](https://www.elastic.co/guide/en/elasticsearch/reference/8.6/index-modules.html#index-modules-settings) in a Dict for creating the index    | dict  |
| `index_mappings`  | Other [index mappings](https://www.elastic.co/guide/en/elasticsearch/reference/8.6/mapping.html) in a Dict for creating the index | dict  |

### RuntimeConfig