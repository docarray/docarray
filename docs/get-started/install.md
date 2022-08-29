# Install

<!-- start frontpage-install -->
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

````{tab} Common install

```bash
pip install "docarray[common]"
```

The following dependencies will be installed to enable the most common features:

| Package | Used in |
|---|---|
| `protobuf` | advanced serialization |
| `lz4` | compression in seralization |
| `requests` | push/pull to Jina Cloud |
| `matplotlib` | visualizing image sprites |
| `Pillow` | image data-related IO |
| `fastapi`| used in embedding projector of DocumentArray|
| `uvicorn`| used in embedding projector of DocumentArray|

````

````{tab} Full install

```bash
pip install "docarray[full]"
```

In addition to `common`, the following dependencies will be installed to enable full features:

| Package | Used in |
|---|---|
| `scipy` | for sparse embedding, tensors |
| `av` | for video processing and IO |
| `trimesh` | for 3D mesh processing and IO |
| `strawberry-graphql` | for GraphQL support |

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


<!-- end frontpage-install -->


## On Apple Silicon

If you own a MacOS device with an Apple Silicon M1/M2 chip, you can run DocArray **natively** on it (instead of running under Rosetta) and enjoy much better performance. This section summarizes how to install DocArray on Apple Silicon device.

### Check terminal and device

To make sure you are using the right terminal, run

```bash
uname -m
```

and it should return

```text
arm64
```


### Install Homebrew

`brew` is a package manager for macOS. If you already install it you need to confirm it is actually installed for Apple Silicon not for Rosetta. To check that, run

```bash
which brew
```

```text
/opt/homebrew/bin/brew
```

If you find it is installed under `/usr/local/` instead of `/opt/homebrew/`, it means your `brew` is installed for Rosetta not for Apple Silicon. You need to reinstall it. [Here is an article on how to do it](https://apple.stackexchange.com/a/410829).

```{danger}
Reinstalling `brew` can be a destructive operation. Please make sure you have backed up your data before proceeding.
```

To (re)install brew, run

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

You may want to observe the output to check if it contains `/opt/homebrew` to make sure you are installing for Apple Silicon.

### Install Python

Python has to be installed natively for Apple Silicon as well. It is possible it is installed for Rosetta, and you are not aware of that. To confirm, run

```python
import platform

platform.machine()
```

which should give

```text
'arm64'
```

If not, then you are using Python under Rosetta, and you need to install Python for Apple Silicon with `brew`.


```bash
brew install python3
```

As of Aug 2022, this will install Python 3.10 natively for Apple Silicon.

Make sure to note down where `python` and `pip` are installed to. In this example, they are installed to `/opt/homebrew/bin/python3` and `/opt/homebrew/opt/python@3.10/libexec/bin/pip` respectively.

### Install dependencies wheels

There are some core dependencies that DocArray needs to run, whose wheels are not available on PyPI but fortunately are available on wheel. To install them, run

```bash
brew install protobuf numpy
```

### Install DocArray

Now we can install Jina via `pip`. Note you need to use the right one:

```bash
/opt/homebrew/opt/python@3.10/libexec/bin/pip install docarray
```


Congratulations! You have successfully installed Jina on Apple Silicon.


````{tip}

To install MPS-enabled PyTorch, run

```bash
/opt/homebrew/opt/python@3.10/libexec/bin/pip install -U --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```
````




