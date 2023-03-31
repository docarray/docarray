# User Guide - Intro

This user guide shows you how to use `DocArray` with most of its features.

They are three main sections:

- [Representing Data](representing/first_step.md): This section will show you how to use `DocArray` to represent your data.
- [Sending Data](sending/first_step.md): This section will show you how to use `DocArray` to send your data.
- [Storing Data](storing/first_step.md): This section will show you how to use `DocArray` to store your data.

You should first start by reading the [Representing Data](representing/first_step.md) section and both the [Sending Data](sending/first_step.md) and [Storing Data](storing/first_step.md) sections can be read in any order.

You wil first need to install `DocArray` in you python environment. 
## Install DocArray

To install `DocArray` to follow this user guide, you can use the following command:

```console
pip install "docarray[full]"
```

This will install the main dependencies of `DocArray` and will work will all the modalities supported.


!!! note 
    To install a very light version of `DocArray` with only the core dependencies, you can use the following command:
    ```
    pip install "docarray"
    ``` 
    
    If you want to use protobuf and DocArray you can do

    ```
    pip install "docarray[proto]"
    ``` 

Depending on your usage you might want to only use `DocArray` with only a couple of specific modalities. 
For instance let's say you only want to work with images, you can install `DocArray` using the following command:

```
pip install "docarray[image]"
```

or with image and audio


```
pip install "docarray[image, audio]"
```

!!! warning 
    This way of installing `DocArray` is only valid starting with version `0.30`