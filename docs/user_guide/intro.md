# Introduction

This user guide shows you how to use `DocArray` with most of its features.

There are three main sections:

- [Representing data](representing/first_step.md): This section will show you how to represent your data. This is a great starting point if you want to better organize the data in your ML models, or if you are looking for a "Pydantic for ML".
- [Sending data](sending/first_step.md): This section will show you how to send your data. This is a great starting point if you want to serve your ML model, for example through FastAPI.
- [Storing data](storing/first_step.md): This section will show you how to store your data. This is a great starting point if you are looking for an "ORM for vector databases".

You should start by reading the [Representing data](representing/first_step.md) section, and then the [Sending data](sending/first_step.md) and [Storing data](storing/first_step.md) sections can be read in any order.

You will first need to install `DocArray` in your Python environment. 

## Install DocArray

To install `DocArray`, you can use the following command:

```console
pip install "docarray[full]"
```

This will install the main dependencies of `DocArray` and will work with all the supported data modalities.

!!! note 
    To install a very light version of `DocArray` with only the core dependencies, you can use the following command:
    ```
    pip install "docarray"
    ``` 
    
    If you want to use `protobuf` and `DocArray`, you can run:

    ```
    pip install "docarray[proto]"
    ``` 

Depending on your usage you might want to use `DocArray` with only a couple of specific modalities and their dependencies. 
For instance, if you only want to work with images, you can install `DocArray` using the following command:

```
pip install "docarray[image]"
```

...or with images and audio:

```
pip install "docarray[image, audio]"
```
