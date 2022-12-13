(fastapi-support)=
# FastAPI/Pydantic

DocArray supports the [pydantic data model](https://pydantic-docs.helpmanual.io/) via {class}`~docarray.document.pydantic_model.PydanticDocument` and {class}`~docarray.document.pydantic_model.PydanticDocumentArray`.
Let's take a look at what this means.

When you want to send or receive Document or DocumentArray objects via the REST API, you can use {meth}`~docarray.array.mixins.io.json.JsonIOMixin.from_json`/{meth}`~docarray.array.mixins.io.json.JsonIOMixin.to_json` to convert the Document or DocumentArray object into JSON. This has been introduced in the {ref}`Serialization<docarray-serialization>` section.

This method, although quite intuitive to many data scientists, is *not* the modern way of building API services. Your engineer friends won't be happy if you give them a service like this. The main problem is **data validation**.

Of course, you can include data validation in your service logic, but this can be difficult as you need to check field by field and repeat things like `isinstance(field, int)`, not to mention handling nested JSON.

Modern web frameworks validate the data _before_ it enters the core logic. For example, [FastAPI](https://fastapi.tiangolo.com/) leverages [pydantic](https://pydantic-docs.helpmanual.io/) to validate input and output data.

This chapter introduces how to use DocArray's pydantic support in a FastAPI service to build a modern API service. The fundamentals of FastAPI can be learned from its docs. I won't repeat them here again.

```{tip}
Features introduced in this chapter require `fastapi` and `pydantic` as dependencies. Please run `pip install "docarray[full]"` to enable it.
```

(schema-gen)=
## JSON Schema

You can get the [JSON Schema](https://json-schema.org/) (OpenAPI itself is based on JSON Schema) of a Document or DocumentArray using the {meth}`~docarray.array.mixins.pydantic.PydanticMixin.get_json_schema` method.

````{tab} Document
```python
from docarray import Document
Document.get_json_schema()
```

```json
{
  "$ref": "#/definitions/PydanticDocument",
  "definitions": {
    "PydanticDocument": {
      "title": "PydanticDocument",
      "type": "object",
      "properties": {
        "id": {
          "title": "Id",
          "type": "string"
        },
```
````
````{tab} DocumentArray
```python
from docarray import DocumentArray
DocumentArray.get_json_schema()
```

```json
{
  "title": "DocumentArray Schema",
  "type": "array",
  "items": {
    "$ref": "#/definitions/PydanticDocument"
  },
  "definitions": {
    "PydanticDocument": {
      "title": "PydanticDocument",
      "type": "object",
      "properties": {
        "id": {
          "title": "Id",
```
````

When you give this schema to your engineer friends, they'll be able to understand the data format you're working with. These schemas also help them to easily integrate DocArray into any webservice.


## Validate incoming Document and DocumentArray objects

You can import {class}`~docarray.document.pydantic_model.PydanticDocument` and {class}`~docarray.document.pydantic_model.PydanticDocumentArray` pydantic data models, and use them to type hint your endpoint. This enables data validation.

```python
from docarray.document.pydantic_model import PydanticDocument, PydanticDocumentArray
from fastapi import FastAPI

app = FastAPI()

@app.post('/single')
async def create_item(item: PydanticDocument):
    ...

@app.post('/multi')
async def create_array(items: PydanticDocumentArray):
    ...
```

Let's now send some JSON:

```python
from starlette.testclient import TestClient
client = TestClient(app)

response = client.post('/single', {'hello': 'world'})
print(response, response.text)
response = client.post('/single', {'id': [12, 23]})
print(response, response.text)
```

```text
<Response [422]> {"detail":[{"loc":["body"],"msg":"value is not a valid dict","type":"type_error.dict"}]}
<Response [422]> {"detail":[{"loc":["body"],"msg":"value is not a valid dict","type":"type_error.dict"}]}
```

Both got rejected (422 error) as they are not valid.

## Convert between pydantic model and DocArray objects

{class}`~docarray.document.pydantic_model.PydanticDocument` and {class}`~docarray.document.pydantic_model.PydanticDocumentArray` are mainly for data validation. When you want to implement real logic, you need to convert them into a Document or DocumentArray. This can be easily achieved via the {meth}`~docarray.array.mixins.pydantic.PydanticMixin.from_pydantic_model` method. When you are done with processing and want to send a {class}`~docarray.document.pydantic_model.PydanticDocument` back, you can call {meth}`~docarray.array.mixins.pydantic.PydanticMixin.to_pydantic_model`.

In a nutshell, the whole procedure looks like the following:

```{figure} lifetime-pydantic.svg
```


Let's see an example:

```python
from docarray import Document, DocumentArray

@app.post('/single')
async def create_item(item: PydanticDocument):
    d = Document.from_pydantic_model(item)
    # now `d` is a Document object
    ...  # process `d` how ever you want
    return d.to_pydantic_model()
    

@app.post('/multi')
async def create_array(items: PydanticDocumentArray):
    da = DocumentArray.from_pydantic_model(items)
    # now `da` is a DocumentArray object
    ...  # process `da` how ever you want
    return da.to_pydantic_model()
```



## Limit returned fields by response model

Supporting the pydantic data model means much more than data validation. One useful pattern is to define a smaller data model and restrict the response to certain fields of the Document.

Imagine we have a DocumentArray with `.embeddings` on the server side, but we do not want to return them to the client for various reasons (for example, they may be meaningless to users, or too big to transfer). You can simply define the fields of interest via 
 `pydantic.BaseModel` and then use it in `response_model=`.

```python
from pydantic import BaseModel
from docarray import Document

class IdOnly(BaseModel):
    id: str

@app.get('/single', response_model=IdOnly)
async def get_item_no_embedding():
    d = Document(embedding=[1, 2, 3])
    return d.to_pydantic_model()
```

And you get:

```text
<Response [200]> {'id': '065a5548756211ecaa8d1e008a366d49'}
```

## Limit returned results recursively

The same idea applies to DocumentArray as well. Say after [`.match()`](../documentarray/matching.md), you are only interested in `.id`—the parent `.id` and the `.id`s of all matches. You can declare a `BaseModel` as follows:

```python
from typing import List, Optional
from pydantic import BaseModel

class IdAndMatch(BaseModel):
    id: str
    matches: Optional[List['IdAndMatch']]
```

Bind it to `response_model`:

```python
@app.get('/get_match', response_model=List[IdAndMatch])
async def get_match_id_only():
    da = DocumentArray.empty(10)
    da.embeddings = np.random.random([len(da), 3])
    da.match(da)
    return da.to_pydantic_model()
```

Then you get a very nice result containing `id`s of matches (of potentially unlimited depth). 

```text
[{'id': 'ef82e4f4756411ecb2c01e008a366d49',
  'matches': [{'id': 'ef82e4f4756411ecb2c01e008a366d49', 'matches': None},
              {'id': 'ef82e6d4756411ecb2c01e008a366d49', 'matches': None},
              {'id': 'ef82e760756411ecb2c01e008a366d49', 'matches': None},
              {'id': 'ef82e7ec756411ecb2c01e008a366d49', 'matches': None},
              ...
```

If `'matches': None` is annoying to you (they are here because you didn't compute second-degree matches), you can further leverage FastAPI's features and do:

```python
@app.get('/get_match', response_model=List[IdMatch], response_model_exclude_none=True)
async def get_match_id_only():
    ...
```

Finally, you get a very clean result with `id`s and matches only:

```text
[{'id': '3da6383e756511ecb7cb1e008a366d49',
  'matches': [{'id': '3da6383e756511ecb7cb1e008a366d49'},
              {'id': '3da63a14756511ecb7cb1e008a366d49'},
              {'id': '3da6392e756511ecb7cb1e008a366d49'},
              {'id': '3da63b72756511ecb7cb1e008a366d49'},
              {'id': '3da639ce756511ecb7cb1e008a366d49'},
              {'id': '3da63a5a756511ecb7cb1e008a366d49'},
              {'id': '3da63ae6756511ecb7cb1e008a366d49'},
              {'id': '3da63aa0756511ecb7cb1e008a366d49'},
              {'id': '3da63b2c756511ecb7cb1e008a366d49'},
              {'id': '3da63988756511ecb7cb1e008a366d49'}]},
 {'id': '3da6392e756511ecb7cb1e008a366d49',
  'matches': [{'id': '3da6392e756511ecb7cb1e008a366d49'},
              {'id': '3da639ce756511ecb7cb1e008a366d49'},
              ...
```

For more information about [pydantic](https://pydantic-docs.helpmanual.io/) and [FastAPI](https://fastapi.tiangolo.com/), we strongly recommend reading their documentation.
