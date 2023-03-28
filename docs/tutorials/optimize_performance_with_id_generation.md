# How to optimize performance

### `BaseDoc`'s id

DocArray's `BaseDoc` has an optional `id` field, which defaults to `ID(os.urandom(16).hex())`. This takes quite some time.
If you don't rely on the id anywhere, you can instead set the default to None. This increases the performance by a factor of approximately 1.4.

```python
from docarray import BaseDoc
from docarray.typing import ID


class MyDoc(BaseDoc):
    id: ID = None
    title: str
```

Since the `BaseDoc.id` is optional, you could also set the value to None, but this turns out to be a bit less efficient than the option above, and increases the performance by a factor of approximately 1.2.

```python
class MyDoc2(BaseDoc):
    title: str


doc = MyDoc2(id=None, title='bye')
```
