# Welcome to DocArray!

```{include} ../README.md
:start-after: <!-- start elevator-pitch -->
:end-before: <!-- end elevator-pitch -->
```

## Install

![PyPI](https://img.shields.io/pypi/v/docarray?color=%23ffffff&label=%20) is the latest version.

Make sure you have Python 3.7+ and `numpy` installed on Linux/Mac/Windows:

````{tab} Basic install

```bash
pip install docarray
```

No extra dependency will be installed.
````

````{tab} Basic install via Conda

```bash
conda install -c conda-forge docarray
```

No extra dependency will be installed.
````

````{tab} Full install

```bash
pip install "docarray[full]"
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

````{tab} Developement install

```bash
pip install "docarray[full,test]"
```

This will install all requirements for reproducing tests on your local dev environment.
````


```pycon
>>> import docarray
>>> docarray.__version__
'0.1.0'
>>> from docarray import Document, DocumentArray
```




```{important}
Jina 3.x[^1] users do not need to install `docarray` separately, as it is shipped with Jina. To check your Jina version, type `jina -vf` in the console.

However, if the printed version is smaller than `0.1.0`, say `0.0.x`, then you are 
not installing `docarray` correctly. You are probably still using an old `docarray` shipped with Jina 2.x. 
```

[^1]: Jina 3.0rc will be released in Feb. 2022. Stay tune!


```{include} ../README.md
:start-after: <!-- start support-pitch -->
:end-before: <!-- end support-pitch -->
```

```{toctree}
:caption: Get Started
:hidden:

get-started/what-is
```

```{toctree}
:caption: User Guides
:hidden:

fundamentals/document/index
fundamentals/documentarray/index
datatypes/index

```

```{toctree}
:caption: Integrations
:hidden:

fundamentals/notebook-support/index
fundamentals/fastapi-support/index
advanced/graphql-support/index
advanced/document-store/index
```


```{toctree}
:caption: Developer References
:hidden:
:maxdepth: 1

api/docarray
proto/index
changelog/index
advanced/document-store/extend
```


---
{ref}`genindex` | {ref}`modindex`

