# Glossary

DocArray scope is a edge of different field, from AI to web app. To make it easier to understand, we have created a glossary of terms used in the documentation. 


## Concept

### `Multi Modal Data`
Multi Modal data is data that is composed of different modalities, Image, Text, Video, Audio, etc.
For example, a YouTube video is composed of a video, a title, a description, a thumbnail, etc. 

Actually most of the data we have in the world is multi-modal.

### `Multi Modal AI`

Multi Modal AI is the field of AI that focus on multi-modal data. 

Most of the recent breakthrough in AI are actually multi-modal AI. 

* [StableDiffusion](https://stability.ai/blog/stable-diffusion-public-release), [MidJourney](https://www.midjourney.com/home/?callbackUrl=%2Fapp%2F), [Dalle-2](https://openai.com/product/dall-e-2) generate *image* from *text*.
* [Whisper](https://openai.com/research/whisper) can generate *text* from *speech* 
* [GPT4](https://openai.com/product/gpt-4), [Flamingo](https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model) are MLLM (Multi Modal Large language Model) that can undersrtand both *text* and *image*.
* ...

One of the reason that AI lab are focusing on multi-modal AI is that is can solve a lot of practical problem and that is actually might be
a requirement to build strong AI system as argued by Yann Lecun in [this article](https://www.noemamag.com/ai-and-the-limits-of-language/) where he said that `A system trained on language alone will never approximate human intelligence`.

### `Generative AI`

Generative AI is as well in the epicenter of the latest AI revolution. These tool allow to *generate* data.

* [StableDiffusion](https://stability.ai/blog/stable-diffusion-public-release), [MidJourney](https://www.midjourney.com/home/?callbackUrl=%2Fapp%2F), [Dalle-2](https://openai.com/product/dall-e-2) generate *image* from *text*.


### `Neural Search`

Neural search is search powered by neural network.  Unlike traditional keyword-based search methods, neural search can understand the context and semantic meaning of the query, allowing it to find relevant results even when the exact keywords are not present


### `Vector Database`

A vector database is a specialized storage system designed to handle high-dimensional vectors, which are common representations of data in machine learning and AI applications. It enables efficient storage, indexing, and querying of these vectors, and typically supports operations like nearest neighbor search, similarity search, and clustering


## Tools

### `Jina`

[Jina](https://jina.ai) is a framework to build Multi Modal application. It heavily relies on DocArray to represent and send data.

Originally DocArray was part of Jina but it became a standalone project that is now independent of Jina.

### `Pydantic`

[Pydantic](https://github.com/pydantic/pydantic/) is a python library that allow to data validation using Python type hints. 
DocArray relies on Pydantic.

### `FastAPI`

[FastAPI](https://fastapi.tiangolo.com/) is a python library that allow to build API using Python type hints.

It is build on top of Pydantic and nicely extend to DocArray

### `Weaviate`

[Weaviate](https://weaviate.io/) is an open-source vector database that is supported in DocArray

### `Weaviate`

[Qdrant](https://qdrant.tech/) is an open-source vector database that is supported in DocArray