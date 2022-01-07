(documentarray)=
# DocumentArray

{class}`~docarray.array.document.DocumentArray` is a list-like container of {class}`~docarray.document.Document` objects. It is **the best way** when working with multiple Documents.

In a nutshell, you can simply consider it as a Python `list`, as it implements **all** list interfaces. That is, if you know how to use Python `list`, you already know how to use DocumentArray. 

It is also powerful as Numpy `ndarray`, allowing you to access elements and attributes by {ref}`fancy slicing syntax<access-elements>`. 

What makes it more exciting is those advanced features of DocumentArray. These features greatly facilitate data scientists work on accessing nested elements, evaluating, visualizing, parallel computing, serializing, matching etc. 

## What's next?

Let's see how to construct a DocumentArray {ref}`in the next section<construct-array>`.

```{toctree}
:hidden:

construct
serialization
access-elements
access-attributes
embedding
matching
evaluation
parallelization
visualization
```