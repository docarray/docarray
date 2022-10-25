# Notebook/Colab Support

Many data scientists work with Jupyter Notebook or Google Colab especially at the early prototyping stage. We understand that and that's why we optimize DocArray's user experience in these two environments. In this chapter, I use Jupyter Notebook as an example to demonstrate the features that can improve your productivity. On Google Colab, it is the same experience.

```{tip}
Some features require extra dependencies beyond the basic install of DocArray. Use `pip install "docarray[full]"` to enable them.
```

## Display Document

A cell with a Document object will be pretty-printed with its non-empty field and `id`.

```{figure} single-doc.png
```

If a Document is nested, then it pretty-prints the nested structure.

```{figure} single-doc-nested.png
```

### Display rich content

If a Document is an image Document, you can use {meth}`~docarray.document.mixins.plot.PlotMixin.display` to visualize it.

```{figure} image-doc.png
```

Note that it finds `.tensor` or `.uri` for visualization.

```{figure} image-doc-blob.png
```

This works even if your Document is not a real image but just a `ndarray` in `.tensor`.

```{figure} image-blob.png
```

Video and audio Document can be displayed as well, you can play them in the cell.

```{figure} audio-video.png
```

## Display DocumentArray

A cell with a DocumentArray object can be pretty-printed automatically.

```{figure} doc-array.png
```

### Display image sprite

DocumentArray with all image Documents (image is either in `.uri` or `.tensor`) can be plotted into one sprite image.

```{figure} image-sprite.png
```

### Display embeddings

DocumentArray with non-empty `.embeddings` can be visualized interactively via {meth}`~docarray.array.mixins.plot.PlotMixin.plot_embeddings`

```{figure} embedding-ani1.gif
```


DocumentArray with non-empty `.embeddings`  and image Documents can be visualized in a much richer way via `.plot_embeddings(image_sprites=True)`.

```{figure} embedding-ani2.gif
```

