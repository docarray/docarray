# Construct

You can construct a `DocumentArray` in different ways:

````{tab} From empty Documents
```python
from jina import DocumentArray

da = DocumentArray.empty(10)
```
````
````{tab} From list of Documents
```python
from jina import DocumentArray, Document

da = DocumentArray([Document(...), Document(...)])
```
````
````{tab} From generator
```python
from jina import DocumentArray, Document

da = DocumentArray((Document(...) for _ in range(10)))
```
````
````{tab} From another DocumentArray
```python
from jina import DocumentArray, Document

da = DocumentArray((Document() for _ in range(10)))
da1 = DocumentArray(da)
```
````

````{tab} From JSON, CSV, ndarray, files, ...

You can find more details about those APIs in {class}`~jina.types.arrays.mixins.io.from_gen.FromGeneratorMixin`.

```python
da = DocumentArray.from_ndjson(...)
da = DocumentArray.from_csv(...)
da = DocumentArray.from_files(...)
da = DocumentArray.from_lines(...)
da = DocumentArray.from_ndarray(...)
```
````
