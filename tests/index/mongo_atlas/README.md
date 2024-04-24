# Setup of Atlas Required

To run Integration tests, one will first need to create the following **Collections** and **Search Indexes**
with the `MONGODB_DATABASE` in the cluster connected to with your `MONGODB_URI`.

Instructions of how to accomplish this in your browser are given in
`docs/API_reference/doc_index/backends/mongodb.md`.

 
Below is the mapping of collections to indexes along with their definitions.

| Collection                | Index Name     | JSON Definition    |     Tests
|---------------------------|----------------|--------------------|---------------------------------|
| simpleschema              | vector_index   | [1]                | test_filter,test_find,test_index_get_del, test_persist_data,  test_text_search |
| mydoc__docs               | vector_index   | [2]                |      test_subindex      |
| mydoc__list_docs__docs    | vector_index   | [3]                |      test_subindex      |
| flatschema                | vector_index_1 | [4]                |      test_find          |
| flatschema                | vector_index_2 | [5]                |      test_find          |
| nesteddoc                 | vector_index_1 | [6]                |      test_find      |
| nesteddoc                 | vector_index   | [7]                |      test_find      |
| simpleschema              | text_index     | [8]                |      test_text_search      |


And here are the JSON definition references:

[1] Collection: `simpleschema` Index name: `vector_index`
```json
{
  "fields": [
    {
      "numDimensions": 10,
      "path": "embedding",
      "similarity": "cosine",
      "type": "vector"
    },
    {
      "path": "number",
      "type": "filter"
    },
    {
      "path": "text",
      "type": "filter"
    }
  ]
}
```

[2] Collection: `mydoc__docs` Index name: `vector_index`
```json
{
  "fields": [
    {
      "numDimensions": 10,
      "path": "simple_tens",
      "similarity": "euclidean",
      "type": "vector"
    }
  ]
}
```

[3] Collection: `mydoc__list_docs__docs` Index name: `vector_index`
```json
{
  "fields": [
    {
      "numDimensions": 10,
      "path": "simple_tens",
      "similarity": "euclidean",
      "type": "vector"
    }
  ]
}
```

[4] Collection: `flatschema` Index name: `vector_index_1`
```json
{
  "fields": [
    {
      "numDimensions": 10,
      "path": "embedding1",
      "similarity": "cosine",
      "type": "vector"
    }
  ]
}
```

[5] Collection: `flatschema` Index name: `vector_index_2`
```json
{
  "fields": [
    {
      "numDimensions": 50,
      "path": "embedding2",
      "similarity": "cosine",
      "type": "vector"
    }
  ]
}
```

[6] Collection: `nesteddoc` Index name: `vector_index_1`
```json
{
  "fields": [
    {
      "numDimensions": 10,
      "path": "d__embedding",
      "similarity": "cosine",
      "type": "vector"
    }
  ]
}
```

[7] Collection: `nesteddoc` Index name: `vector_index`
```json
{
  "fields": [
    {
      "numDimensions": 10,
      "path": "embedding",
      "similarity": "cosine",
      "type": "vector"
    }
  ]
}
```

[8] Collection: `simpleschema` Index name: `text_index`
 
```json
{
  "mappings": {
    "dynamic": false,
    "fields": {
      "text": [
        {
          "type": "string"
        }
      ]
    }
  }
}
```

NOTE: that all but this final one (8) are Vector Search Indexes. 8 is a Text Search Index.


With these in place you should be able to successfully run all of the tests as follows. 

```bash
MONGODB_URI=<uri> MONGODB_DATABASE=<db_name> py.test tests/index/mongo_atlas/
```

IMPORTANT: FREE clusters are limited to 3 search indexes. 
As such, you may have to (re)create accordingly.