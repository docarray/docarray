# Visualization

If you have an image Document (with possible image data in `.uri`/`.tensor`), you can directly visualize it via {meth}`~docarray.document.mixins.plot.PlotMixin.display`.

```{figure} images/doc-plot-in-jupyter.png
```


To better see the Document's nested structure, you can use {meth}`~docarray.document.mixins.plot.PlotMixin.summary`.

```{code-block} python
---
emphasize-lines: 13,14
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

```text
 <Document ('id', 'embedding', 'chunks', 'matches') at 🐲>
    └─ matches
          └─ <Document ('id', 'adjacency', 'embedding') at 🐯>
    └─ chunks
          └─ <Document ('id', 'parent_id', 'granularity', 'embedding', 'chunks') at 🐦>
              └─ chunks
                    └─ <Document ('id', 'parent_id', 'granularity', 'embedding') at 🐢>
```

When using Notebook/Colab, this is auto-rendered.

```{figure} images/doc-auto-summary.png
```
