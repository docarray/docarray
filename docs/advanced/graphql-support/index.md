# GraphQL


DocArray supports GraphQL. You can use GraphQL to query a DocumentArray and get exactly the fields you need: `.embedding` is too big and too verbose, then don't query it. Comparing to the REST API, clients using GraphQL are fast and stable because they control the data they get, not the server. 

When integrating DocArray in a GraphQL app, you only need to implement the *query* (in GraphQL idiom, this is like the API endpoint that your server allows). The *schema* part is provided by DocArray and can be used out of the box.


````{tip}
This feature requires `strawberry`. You can install it via `pip install "docarray[full]"` or `pip install "strawberry-graphql[debug-server]"`. 
````

```{seealso}
This article does *not* serve as the introduction to GraphQL. If you don't have GraphQL background, it is stronly recommended to learn more about GraphQL in the [official GraphQL documentation](https://graphql.org/). You may also want to learn more about [Strawberry](https://strawberry.rocks/). Otherwise, you may get confused by the GraphQL idioms, e.g. query, schema.
```

## Basic example

Let's create dummy matches in a DocumentArray:

```python
from docarray import DocumentArray
import numpy as np

da = DocumentArray.empty(3)
da.embeddings = np.random.random([3, 15])

db = DocumentArray.empty(4)
db.embeddings = np.random.random([4, 15])

da.match(db)
da.summary()
```

```text
                     Documents Summary                      
                                                            
  Length                    3                               
  Homogenous Documents      True                            
  Has nested Documents in   ('matches',)                    
  Common Attributes         ('id', 'embedding', 'matches')  
                                                            
                        Attributes Summary                        
                                                                  
  Attribute   Data type         #Unique values   Has empty value  
 ──────────────────────────────────────────────────────────────── 
  embedding   ('ndarray',)      3                False            
  id          ('str',)          3                False            
  matches     ('MatchArray',)   3                False            
                                                                  
          Storage Summary          
                                   
  Class     DocumentArrayInMemory  
  Backend   In Memory              
```


Now let's build a *query* (remember in GraphQL this means an endpoint) that allows users to fetch this DocumentArray:

```python
from typing import List
from docarray.document.strawberry_type import StrawberryDocument
import strawberry

@strawberry.type
class Query:
    docs: List[StrawberryDocument] = strawberry.field(
        resolver=lambda: da.to_strawberry_type()
    )

schema = strawberry.Schema(query=Query)
``` 

Notice how I leverage {class}`~docarray.document.strawberry_type.StrawberryDocument` and use {meth}`~docarray.array.mixins.pydantic.StrawberryMixin.to_strawberry_type` to convert the type in the resolver before returning the result.

In practice, `da` could be your final search results, or some DocumentArray after embedding or preprocessing. Here I just use the dummy matches I created before to serve as the results.

Finally, save all code snippets above into `toy.py` and run it from the console via:

```bash
strawberry server toy
```

You will see 
```text
Running strawberry on http://0.0.0.0:8000/graphql 🍓
```


Now open `http://0.0.0.0:8000/graphql` in your browser. You should be able to see a GraphiQL playground at this url.

Try the following query
```gql
{
    docs {
        id
    }
}
```

```{figure} gql-ui.png
```

Now we have one endpoint that allows user to selectively read fields from a DocumentArray. Additional endpoints can be added to `Query` class, to support advance filtering and selecting, but this is beyond the scope of this tutorial. It is also your responsibility as the app/service provider to decide what API you want to expose to users. 

## Integrate with FastAPI

Strawberry's built-in server is perfect for prototyping an API. When it comes to production, you can use FastAPI. Here is a short example how you can wrap the above snippet it in a FastAPI app:

```python
from strawberry.asgi import GraphQL
from fastapi import FastAPI

graphql_app = GraphQL(schema)

app = FastAPI()
app.add_route('/graphql', graphql_app)
app.add_websocket_route('/graphql', graphql_app)
```

You can learn more about [FastAPI GraphQL support from here](https://fastapi.tiangolo.com/advanced/graphql/).

