# Glossary

DocArray's scope is at the edge of different fields, from AI to web apps. To make it easier to understand, we have created a glossary of terms used in the documentation. 

## Concept

### `Multimodal Data`
Multimodal data is data that is composed of different modalities, like Image, Text, Video, Audio, etc.
For example, a YouTube video is composed of a video, a title, a description, a thumbnail, etc. 

Actually, most of the data we have in the world is multimodal.

### `Multimodal AI`

Multimodal AI is the field of AI that focuses on multimodal data. 

Most of the recent breakthroughs in AI are multimodal AI. 

* [StableDiffusion](https://stability.ai/blog/stable-diffusion-public-release), [Midjourney](https://www.midjourney.com/home/?callbackUrl=%2Fapp%2F), [DALL-E 2](https://openai.com/product/dall-e-2) generate *images* from *text*.
* [Whisper](https://openai.com/research/whisper) generates *text* from *speech*.
* [GPT-4](https://openai.com/product/gpt-4) and [Flamingo](https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model) are MLLMs (Multimodal Large Language Models) that understand both *text* and *images*.

One of the reasons that AI labs are focusing on multimodal AI is that it can solve a lot of practical problems and that it actually might be
a requirement to build a strong AI system as argued by Yann Lecun in [this article](https://www.noemamag.com/ai-and-the-limits-of-language/) where he stated that "a system trained on language alone will never approximate human intelligence."

### `Generative AI`

Generative AI is also in the epicenter of the latest AI revolution. These tools allow us to *generate* data.

* [StableDiffusion](https://stability.ai/blog/stable-diffusion-public-release), [MidJourney](https://www.midjourney.com/home/?callbackUrl=%2Fapp%2F), [Dalle-2](https://openai.com/product/dall-e-2) generate *images* from *text*.
* LLM: Large Language Model, (GPT, Flan, LLama, Bloom). These models generate *text*.

### `Neural Search`

Neural search is search powered by neural networks. Unlike traditional keyword-based search methods, neural search understands the context and semantic meaning of a user's query, allowing it to find relevant results even when the exact keywords are not present.

### `Vector Database`

A vector database is a specialized storage system designed to handle high-dimensional vectors, which are common representations of data in machine learning and AI applications. It enables efficient storage, indexing, and querying of these vectors, and typically supports operations like nearest neighbor search, similarity search, and clustering.

## Tools

### `Jina`

[Jina](https://jina.ai) is a framework to build multimodal applications. It relies heavily on DocArray to represent and send data.

DocArray was originally part of Jina but it became a standalone project that is now independent of Jina.

### `Pydantic`

[Pydantic](https://github.com/pydantic/pydantic/) is a Python library that allows data validation using Python type hints. 
DocArray relies on Pydantic.

### `FastAPI`

[FastAPI](https://fastapi.tiangolo.com/) is a Python library that allows building API using Python type hints.

It is built on top of Pydantic and nicely extends to DocArray.

### `Weaviate`

[Weaviate](https://weaviate.io/) is an open-source vector database that is supported in DocArray.

### `Weaviate`

[Qdrant](https://qdrant.tech/) is an open-source vector database that is supported in DocArray.
