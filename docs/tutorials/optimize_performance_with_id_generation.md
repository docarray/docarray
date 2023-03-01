# How to optimize performance

### `BaseDocument`'s id

DocArray's `BaseDocument` has an optional `id` field, which defaults to `ID(os.urandom(16).hex())`. This takes quite some time.
If you don't rely on the id anywhere, you can instead set the default to None:

```python
from docarray import BaseDocument
from docarray.typing import ID


class MyDoc(BaseDocument):
    id: ID = None
    title: str
```

Since the `BaseDocument.id` is optional, you could also set the value to None, but this turns out to be a bit less efficient than the option above:

```python
class MyDoc2(BaseDocument):
    title: str


doc = MyDoc2(id=None, title='bye')
```

If you do rely on the ID, there is another option to speed up the process. You could use an alternative library to create the ID. 
By default `os.urandom(16).hex()` is being used. You could use e.g. [fastuuid](https://github.com/thedrow/fastuuid)


```python
import fastuuid


class MyDocFastuuid(BaseDocument):
    id: ID = ID(fastuuid.fastuuid.uuid4())
    title: str


doc = MyDocFastuuid(title='bye')
```

Benchmark:
![benchmark_id_generation.png](benchmark_id_generation.png)
