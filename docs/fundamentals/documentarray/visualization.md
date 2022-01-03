# Visualization

If a `DocumentArray` contains all image `Document`, you can plot all images in one sprite image using {meth}`~jina.types.arrays.mixins.plot.PlotMixin.plot_image_sprites`.

```python
from jina import DocumentArray
docs = DocumentArray.from_files('*.jpg')
docs.plot_image_sprites()
```

```{figure} sprite-image.png
:width: 60%
```

(visualize-embeddings)=
If a `DocumentArray` has valid `.embeddings`, you can visualize the embeddings interactively using {meth}`~jina.types.arrays.mixins.plot.PlotMixin.plot_embeddings`.

````{hint}
Note that `.plot_embeddings()` applies to any `DocumentArray` not just image ones. For image `DocumentArray`, you can do one step more to attach the image sprite on to the visualization points.

```python
da.plot_embeddings(image_sprites=True)
```
 
````

```python
import numpy as np
from jina import DocumentArray

docs = DocumentArray.from_files('*.jpg')
docs.embeddings = np.random.random([len(docs), 256])  # some random embeddings

docs.plot_embeddings(image_sprites=True)
```


```{figure} embedding-projector.gif
:align: center
```
