# Visualization

To better see the Document's nested structure, you can use {meth}`~jina.types.document.mixins.plot.PlotMixin.plot` function. If you are using JupyterLab/Notebook,
all `Document` objects will be auto-rendered:


```{code-block} python
---
emphasize-lines: 13
---
import numpy as np
from docarray import Document

d0 = Document(id='🐲', embedding=np.array([0, 0]))
d1 = Document(id='🐦', embedding=np.array([1, 0]))
d2 = Document(id='🐢', embedding=np.array([0, 1]))
d3 = Document(id='🐯', embedding=np.array([1, 1]))

d0.chunks.append(d1)
d0.chunks[0].chunks.append(d2)
d0.matches.append(d3)

d0.summary()
```


```{figure} ../../../.github/images/four-symbol-docs.svg
:align: center
```
