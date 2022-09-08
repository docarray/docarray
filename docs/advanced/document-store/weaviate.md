(weaviate)=
# Weaviate

One can use [Weaviate](https://weaviate.io) as the document store for DocumentArray. It is useful when one wants to have faster Document retrieval on embeddings, i.e. `.match()`, `.find()`. It can store up to millions of data objects. For large-scale implementations, check out the [Weaviate Kubernetes setup](https://weaviate.io/developers/weaviate/current/getting-started/installation.html#kubernetes-k8s).

````{tip}
This feature requires `weaviate-client`. You can install it via `pip install "docarray[weaviate]".` 
````

Here is a video tutorial that guides you to build a simple image search using Weaviate and Docarray.

<center>
<iframe width="560" height="315" src="https://www.youtube.com/embed/rBKvoIGihnY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</center>

## Usage

There are three ways how you can use Weaviate.

* Weaviate Cloud Service
* Docker-compose
* Kubernetes

#### Weaviate Cloud Service

On the [Weaviate Cloud Service](https://console.semi.technology/), you can create a free sandbox to connect to DocArray. When initiating your Weaviate setup, you'll leave the cloud endpoint (see below).

#### Docker-compose

The [configurator](https://weaviate.io/developers/weaviate/current/installation/docker-compose.html#configurator) allows you to get the latest Weaviate version. Make sure to select "Standalone, no modules" in the process.

#### Kubernetes

The Kubernetes setup is a bit more work but comes in handy when scaling your DocArray project to production. All information about running Weaviate with Kubernetes van be found [here](https://weaviate.io/developers/weaviate/current/installation/kubernetes.html).

## Installation

After setting up your Weaviate environment, you can install the dependencies like this:

```bash
pip install -U docarray[weaviate]
docker-compose up
```

### Create DocumentArray with Weaviate backend

Assuming service is started using the default configuration (i.e. server address is `http://localhost:8080` or `https://{unique ID}.semi.network`), one can instantiate a DocumentArray with Weaviate storage as such:

```python
from docarray import DocumentArray

da = DocumentArray(storage='weaviate')
```

The usage would be the same as the ordinary DocumentArray.

You can set the `config={'name': 'SomeValue' })` because Weaviate's class system creates a vector space per class (i.e., the name). You can store documents with or without vectors, but the length of a vector needs to be the same within the class. For multiple embedding sizes the following is perfectly valid:

```
from docarray import DocumentArray

da1 = DocumentArray(storage='weaviate', config={'name': 'Document1'})
da2 = DocumentArray(storage='weaviate', config={'name': 'Document2'})
```

To access a DocumentArray formerly persisted, one can specify the name, the host, the port and the protocol to connect to the server. `name` is required in this case but other connection parameters are optional. If they are not provided, then it will connect to the Weaviate service bound to `http://localhost:8080`.

Note, that the `name` parameter in `config` needs to be capitalized.

```python
from docarray import DocumentArray

da = DocumentArray(
    storage='weaviate', config={'name': 'Document', 'host': 'localhost', 'port': 8080}
)

da.summary()
```

Other functions behave the same as in-memory DocumentArray.

## Config

| Name                       | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Default                                            |
|----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------|
| `host`                     | Hostname of the Weaviate server                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | 'localhost'                                        |
| `port`                     | port of the Weaviate server                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | 8080                                               |
| `protocol`                 | protocol to be used. Can be 'http' or 'https'                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | 'http'                                             |
| `name`                     | Weaviate class name; the class name of Weaviate object to presesent this DocumentArray                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | None                                               |
| `serialize_config`         | [Serialization config of each Document](../../../fundamentals/document/serialization.md)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | None                                               |
| `distance`                 | The distance metric used to compute the distance between vectors. Must be either `cosine` or `l2-squared`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | `None`, defaults to the default value in Weaviate* |
| `ef`                       | The size of the dynamic list for the nearest neighbors (used during the search). The higher ef is chosen, the more accurate, but also slower a search becomes.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | `None`, defaults to the default value in Weaviate* |
| `ef_construction`          | The size of the dynamic list for the nearest neighbors (used during the construction). Controls index search speed/build speed tradeoff.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | `None`, defaults to the default value in Weaviate* |
| `timeout_config`           | Set the timeout configuration for all requests to the Weaviate server.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | `None`, defaults to the default value in Weaviate* |
| `max_connections`          | The maximum number of connections per element in all layers.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | `None`, defaults to the default value in Weaviate* |
| `dynamic_ef_min`           | If using dynamic ef (set to -1), this value acts as a lower boundary. Even if the limit is small enough to suggest a lower value, ef will never drop below this value. This helps in keeping search accuracy high even when setting very low limits, such as 1, 2, or 3.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | `None`, defaults to the default value in Weaviate* |
| `dynamic_ef_max`           | If using dynamic ef (set to -1), this value acts as an upper boundary. Even if the limit is large enough to suggest a lower value, ef will be capped at this value. This helps to keep search speed reasonable when retrieving massive search result sets, e.g. 500+.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | `None`, defaults to the default value in Weaviate* |
| `dynamic_ef_factor`        | If using dynamic ef (set to -1), this value controls how ef is determined based on the given limit. E.g. with a factor of 8, ef will be set to 8*limit as long as this value is between the lower and upper boundary. It will be capped on either end, otherwise.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | `None`, defaults to the default value in Weaviate* |
| `vector_cache_max_objects` | For optimal search and import performance all previously imported vectors need to be held in memory. However, Weaviate also allows for limiting the number of vectors in memory. By default, when creating a new class, this limit is set to 2M objects. A disk lookup for a vector is orders of magnitudes slower than memory lookup, so the cache should be used sparingly.                                                                                                                                                                                                                                                                                                                                                                                                                                | `None`, defaults to the default value in Weaviate* |
| `flat_search_cutoff`       | Absolute number of objects configured as the threshold for a flat-search cutoff. If a filter on a filtered vector search matches fewer than the specified elements, the HNSW index is bypassed entirely and a flat (brute-force) search is performed instead. This can speed up queries with very restrictive filters considerably. Optional, defaults to 40000. Set to 0 to turn off flat-search cutoff entirely.                                                                                                                                                                                                                                                                                                                                                                                           | `None`, defaults to the default value in Weaviate* |
| `cleanup_interval_seconds` | How often the async process runs that “repairs” the HNSW graph after deletes and updates. (Prior to the repair/cleanup process, deleted objects are simply marked as deleted, but still a fully connected member of the HNSW graph. After the repair has run, the edges are reassigned and the datapoints deleted for good). Typically this value does not need to be adjusted, but if deletes or updates are very frequent it might make sense to adjust the value up or down. (Higher value means it runs less frequently, but cleans up more in a single batch. Lower value means it runs more frequently, but might not be as efficient with each run).                                                                                                                                                  | `None`, defaults to the default value in Weaviate* |
| `skip`                     | There are situations where it doesn’t make sense to vectorize a class. For example if the class is just meant as glue between two other class (consisting only of references) or if the class contains mostly duplicate elements (Note that importing duplicate vectors into HNSW is very expensive as the algorithm uses a check whether a candidate’s distance is higher than the worst candidate’s distance for an early exit condition. With (mostly) identical vectors, this early exit condition is never met leading to an exhaustive search on each import or query). In this case, you can skip indexing a vector all-together. To do so, set "skip" to "true". skip defaults to false; if not set to true, classes will be indexed normally. This setting is immutable after class initialization. | `None`, defaults to the default value in Weaviate* |

*You can read more about the HNSW parameters and their default values [here](https://weaviate.io/developers/weaviate/current/vector-index-plugins/hnsw.html#how-to-use-hnsw-and-parameters)

## Filtering on Documents

Search with `.find` can be restricted by user-defined filters. Such filters can be constructed following the guidelines 
in [Weaviate's Documentation](https://weaviate.io/developers/weaviate/current/graphql-references/filters.html).

Add the following data objects to Weaviate to experiment with filtering:

```python
da = DocumentArray(storage='weaviate', config={'name': 'Document'})

d1 = Document(text='Im the nested doc', embedding=[0.1, 0.2, 0.3])
d2 = Document(text='Im the 2nd nested doc', embedding=[0.1, 0.2, 0.3, 0.4])
d3 = Document(text='Im the main doc', chunks=[d1, d2], id="d3")

da.extend(
    [d1, d2, d3]
)
```

### Example of `.find` with a filter

Consider you store Documents with a certain `text` into weaviate and you want to retrieve all Documents 
and you want to retrieve all documents that match the `text` object.

You can filter such documents as follows:

```python
from docarray import Document, DocumentArray

da = DocumentArray(storage='weaviate', config={'name': 'Document'})

# create Documents
d1 = Document(text='Im the nested doc', embedding=[0.1, 0.2, 0.3])
d2 = Document(text='Im the 2nd nested doc', embedding=[0.4, 0.5, 0.6])
d3 = Document(text='Im the main doc', chunks=[d1, d2], id="d3")

# add Documents to Weaviate
da.extend(
    [d1, d2, d3]
)

# filter for text
filter = {'path': 'text', 'operator': 'Equal', 'valueText': 'nested'}
q = da.find(filter=filter, limit=3)

# filter for ID
filter = {'id': 'd3'}
q = da.find(filter=filter)
```

The filters used are based on Weaviate's filtering mechanism, you can learn more about this [here](https://weaviate.io/developers/weaviate/current/graphql-references/filters.html#single-operand).

### Example of `.find` with a vector

Consider Documents with embeddings `[0,0,0]` up to ` [9,9,9]` where the document with embedding `[i,i,i]`.

```python
from docarray import Document, DocumentArray
import numpy as np

from docarray import Document, DocumentArray

n_dim = 3
da = DocumentArray(
    storage='weaviate',
    config={
        'n_dim': n_dim,
        'columns': {'price': 'float'},
    },
)

with da:
    da.extend([Document(id=f'r{i}', tags={'price': i}) for i in range(10)])

print('\nIndexed Prices:\n')
for price in da[:, 'tags__price']:
    print(f'\t price={price}')
```

Then you can retrieve all documents whose price is lower than or equal to `max_price` by applying the following 
filter:

```python
max_price = 3
n_limit = 4

filter = {'path': 'price', 'operator': 'LessThanEqual', 'valueNumber': max_price}
results = da.find(filter=filter)

da = DocumentArray(storage='weaviate', config={'name': 'Document'})

# create Documents
d1 = Document(text='Im the nested doc', embedding=np.random.random([256]))
d2 = Document(text='Im the 2nd nested doc', embedding=np.random.random([256]))
d3 = Document(text='Im the main doc', chunks=[d1, d2], id="d3")

# add Documents to Weaviate
da.extend(
    [d1, d2, d3]
)

# search for a random vector
q = Document(embedding=np.random.random([256]))
q.match(da)
```

### Example of `.find` with a vector and scalar filter

This is the same example as the two examples above, but with the filters mixed.

```python
from docarray import Document, DocumentArray
import numpy as np

from docarray import Document, DocumentArray

da = DocumentArray(storage='weaviate', config={'name': 'Document'})

# create Documents
d1 = Document(text='Im the nested doc', embedding=np.random.random([256]))
d2 = Document(text='Im the 2nd nested doc', embedding=np.random.random([256]))
d3 = Document(text='Im the main doc', chunks=[d1, d2], id="d3")

# add Documents to Weaviate
da.extend(
    [d1, d2, d3]
=======
da = DocumentArray(
    storage='weaviate',
    config={'n_dim': n_dim, 'columns': {'price': 'int'}, 'distance': 'l2-squared'},
)

# Mix the filters
filter = {'path': 'text', 'operator': 'Equal', 'valueText': 'nested'}
embedding = np.random.random([256])
q = da.find(embedding, filter=filter, limit=1)
```

### Example of `.find` for sorting

You can sort results by any primitive property, typically a text, string, number, or int property. When a query has a 
natural order (e.g. because of a near vector search), adding a sort operator will override the order. Further documentation is available [here](https://weaviate.io/developers/weaviate/current/graphql-references/get.html#sorting)

```python
from docarray import Document, DocumentArray
import numpy as np

da = DocumentArray(storage='weaviate', config={'name': 'Document'})

# create Documents
d1 = Document(text='Im the nested doc', granularity=1, embedding=np.random.random([256]))
d2 = Document(text='Im the 2nd nested doc', granularity=2, embedding=np.random.random([256]))
d3 = Document(text='Im the main doc', granularity=3, chunks=[d1, d2], id="d3")

# add Documents to Weaviate
da.extend(
    [d1, d2, d3]
)

# set the filter
filter = {'path': 'text', 'operator': 'Equal', 'valueText': 'nested'}

# define the sorting path
sort = [{'path': ['granularity'], 'order': 'desc'}]

# define the embedding
embedding = np.random.random([256])

# query
q = da.find(embedding, filter=filter, limit=10, sort=sort, query_params={"certainty": 0.9})
```

## Include additional properties in the return

GraphQL additional properties can be used on data objects in Get{} Queries to get additional information about the 
returned data objects. Which additional properties are available depends on the modules that are attached to Weaviate. 
The fields id, certainty, featureProjection and classification are available from Weaviate Core. On nested GraphQL 
fields (references to other data classes), only the id can be returned. Explanation on specific additional properties 
can be found on the module pages, see for example 
[text2vec-contextionary](https://weaviate.io/developers/weaviate/current/modules/text2vec-contextionary.html#additional-graphql-api-properties).

[Further documentation here](https://weaviate.io/developers/weaviate/current/graphql-references/additional-properties.html)

In order to include additional properties on the request you can use the `additional` parameter of the `find()` function.
These will be included as Tags on the response.

Assume you want to know when the document was inserted and last updated in the DB. 
You can run the following:

```python
from docarray import Document, DocumentArray

da = DocumentArray(storage='weaviate', config={'name': 'Document'})

# create Documents
d1 = Document(text='Im the nested doc', embedding=[0.1, 0.2, 0.3])
d2 = Document(text='Im the 2nd nested doc', embedding=[0.4, 0.5, 0.6])
d3 = Document(text='Im the main doc', chunks=[d1, d2], id="d3")

# add Documents to Weaviate
da.extend(
    [d1, d2, d3]
)
=======
# make connection and set columns
da = DocumentArray(
    storage='weaviate',
    config={
        'n_dim': n_dim,
        'columns': {'price': 'float'},
        'distance': 'l2-squared',
        "name": "Persisted",
        "host": "localhost",
        "port": 8080,
    },
)

# load the dummy data
with da:
    da.extend(
        [
            Document(
                text=f'word{i}',
                id=f'r{i}',
                embedding=i * np.ones(n_dim),
                tags={'price': i},
            )
            for i in range(10)
        ]
    )

np_query = np.ones(n_dim) * 8
sort = sort = [{"path": ["price"], "order": "desc"}]  # or "asc"
results = da.find(np_query, sort=sort)

print(
    '\n Returned examples that verify results are in order from highest price to lowest:\n'
)
for embedding, price in zip(results.embeddings, results[:, 'tags__price']):
    print(f'\tembedding={embedding},\t price={price}')
```

# filter for text with additional fields
filter = {'path': 'text', 'operator': 'Equal', 'valueText': 'nested'}
additional = ['creationTimeUnix', 'lastUpdateTimeUnix']
q = da.find(filter=filter, additional=additional)
```

## Example with tokenizer

The following example shows how to use DocArray with Weaviate Document Store in order to index and search text 
Documents.

First, let's run the create the `DocumentArray` instance (make sure a Weaviate server is up and running):

```python
from docarray import DocumentArray

da = DocumentArray(
    storage="weaviate", config={"name": "Document", "host": "localhost", "port": 8080}
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

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')


def collate_fn(da):
    return tokenizer(da.texts, return_tensors='pt', truncation=True, padding=True)


da.embed(model, collate_fn=collate_fn)

results = da.find(
    DocumentArray([Document(text='Persist Documents with Weaviate.')]).embed(
        model,
        collate_fn=collate_fn,
    ),
    query_params={"certainty": 0.9},
)

print("Only results that have a 'weaviate_certainty' of higher than 0.9 should show:")
for res in results:
    print(f"\t text={res[:, 'text']}")
    print(f"\t scores={res[:, 'scores']}")
```

Now, we can generate embeddings inside the database using BERT model:

```python
from docarray import DocumentArray
from docarray import Document
from transformers import AutoModel, AutoTokenizer

def collate_fn(da):
    return tokenizer(da.texts, return_tensors='pt', truncation=True, padding=True)

da = DocumentArray(
    storage="weaviate", config={"name": "Document", "host": "localhost", "port": 8080}
)

da.extend(
    [
        Document(text='Persist Documents with Weaviate.'),
        Document(text='And enjoy fast nearest neighbor search.'),
        Document(text='All while using DocArray API.'),
    ]
)

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

da.embed(model, collate_fn=collate_fn)

results = da.find(
    DocumentArray([Document(text='How to persist Documents')]).embed(
        model, collate_fn=collate_fn
    ),
    limit=1,
)

print(results[0].texts)
=======
print('\n See when the Document was created and updated:\n')
for res in results:
    print(f"\t creationTimeUnix={res[:, 'tags__creationTimeUnix']}")
    print(f"\t lastUpdateTimeUnix={res[:, 'tags__lastUpdateTimeUnix']}")
```

Output:

```text
Persist Documents with Weaviate.
```