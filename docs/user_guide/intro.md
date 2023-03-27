# User Guide - Intro

This user guide show you how to use `DocArray` with most of its features, step by step.

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
    
    If you want to install user protobuf with the minimal dependencies you can do

    ```
    pip install "docarray[common]"
    ``` 

!!! note 
    You can always only install a subset of the dependencies for the modalities that you need.
    For instance lets say you only want to work with images, you can do

    ```
    pip install "docarray[image]"
    ```

    or with image and audio


    ```
    pip install "docarray[image, audio]"
    ```
