# Welcome to DocArray!

```{include} ../README.md
:start-after: <!-- start elevator-pitch -->
:end-before: <!-- end elevator-pitch -->
```

## Install

Make sure you have Python 3.7+ and `numpy` installed on Linux/Mac/Windows:

````{tab} Basic install

```bash
pip install docarray
```

No extra dependency will be installed.
````

````{tab} Full install

```bash
pip install docarray[full]
```

The following dependencies will be installed to enable additional features:

| Package | Used in |
|---|---|
| `protobuf` | advanced serialization |
| `lz4` | compression in seralization |
| `requests` | push/pull to Jina Cloud |
| `matplotlib` | visualizing image sprites |
| `Pillow` | image data-related IO |
| `rich` | push/pull to Jina Cloud, summary of Document, DocumentArray |
| `av` | video data-related IO |
| `trimesh`| 3D mesh data-related IO |
| `fastapi`| used in embedding projector of DocumentArray|

Alternatively, you can first do basic installation and then install missing dependencies on-demand. 
````




```{include} ../README.md
:start-after: <!-- start support-pitch -->
:end-before: <!-- end support-pitch -->
```

```{toctree}
:caption: Fundamentals
:hidden:

fundamentals/document-api.md
fundamentals/documentarray-api.md
```


```{toctree}
:caption: API Reference
:hidden:
:maxdepth: 1

api/docarray
proto/index
```


---
{ref}`genindex` | {ref}`modindex`

