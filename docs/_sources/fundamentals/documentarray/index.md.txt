(documentarray)=
# DocumentArray

This is a Document, we already know it can have different data types and a nested structure:

```{figure} images/docarray-single.svg
:width: 30%
```

This is a DocumentArray:

```{figure} images/docarray-array.svg
:width: 80%
```


A {class}`~docarray.array.document.DocumentArray` is a list-like container of {class}`~docarray.document.Document` objects. It is **the best way** to work with multiple Documents.

In a nutshell, you can simply consider it as a Python `list`, as it implements **all** list interfaces. That is, if you know how to use Python's `list`, you already know how to use DocumentArray.

It is also as powerful as Numpy `ndarray` and Pandas `DataFrame`, letting you efficiently [access elements](access-elements.md) and [attributes](access-attributes.md) of contained Documents.

DocumentArray's advanced features make it even more exciting. These features greatly speed up accessing nested elements, evaluating, visualizing, parallel computing, serializing, matching etc. 

Finally, if your data is too big to fit in memory, you can simply switch to an {ref}`on-disk/remote document store<doc-store>`. The full API and user experience remain the same. There's no need to learn anything else.

## What's next?

Let's see how to construct a DocumentArray {ref}`in the next section<construct-array>`.

```{toctree}
:hidden:

construct
serialization
access-elements
access-attributes
find
parallelization
visualization
post-external
embedding
matching
subindex
evaluation
```
