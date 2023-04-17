# Glossary

DocArray's scope covers several fields, from AI to web apps. To make it easier to understand, we have created a glossary of terms used in the documentation. 

## Concepts

### `Multimodal Data`

Multimodal data is data that is composed of different modalities, like image, text, video, audio, etc.

Actually, most of the data we have in the world is multimodal, for example:

- Newspaper pages are made up of headline, author byline, image, text, etc.
- YouTube videos are made up of a video, title, description, thumbnail, etc. 

### `Multimodal AI`

Multimodal AI is the field of AI that focuses on multimodal data. 

Most of the recent breakthroughs in AI are multimodal AI. 

* [StableDiffusion](https://stability.ai/blog/stable-diffusion-public-release), [Midjourney](https://www.midjourney.com/home/?callbackUrl=%2Fapp%2F) and [DALL-E 2](https://openai.com/product/dall-e-2) generate *images* from *text*.
* [Whisper](https://openai.com/research/whisper) generates *text* from *speech*.
* [GPT-4](https://openai.com/product/gpt-4) and [Flamingo](https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model) are MLLMs (Multimodal Large Language Models) that understand both *text* and *images*.

Many AI labs are focusing on multimodal AI because it can solve a lot of practical problems, and that it might actually be
a requirement for strong AI systems (as argued by Yann Lecun in [this article](https://www.noemamag.com/ai-and-the-limits-of-language/) where he states that "a system trained on language alone will never approximate human intelligence.")

### `Generative AI`

Generative AI is also in the epicenter of the latest AI revolution. These tools allow us to *generate* data.

* [StableDiffusion](https://stability.ai/blog/stable-diffusion-public-release), [MidJourney](https://www.midjourney.com/home/?callbackUrl=%2Fapp%2F), and [Dalle-2](https://openai.com/product/dall-e-2) generate *images* from *text*.
* LLMs: Large Language Models, (GPT, Flan, LLama, Bloom). These models generate *text*.

### `Neural Search`

Neural search is search powered by neural networks. Unlike traditional keyword-based search methods, neural search understands the context and semantic meaning of a user's query, allowing it to find relevant results even when the exact keywords are not present.

### `Vector Database`

A vector database is a specialized storage system designed to handle high-dimensional vectors, which are common representations of data in machine learning and AI applications. It enables efficient storage, indexing, and querying of these vectors, and typically supports operations like nearest neighbor search, similarity search, and clustering.

## Tools

### `Jina`

[Jina](https://github.com/jina-ai/jina/) is a framework for building multimodal applications. It relies heavily on DocArray to represent and send data.

DocArray was originally part of Jina but it is now a standalone project independent of Jina.

### `Pydantic`

[Pydantic](https://github.com/pydantic/pydantic/) is a Python library that allows data validation using Python type hints. 
DocArray relies on Pydantic.

### `FastAPI`

[FastAPI](https://fastapi.tiangolo.com/) is a Python library that allows building API using Python type hints. It is built on top of Pydantic and nicely extends to DocArray.

### `Weaviate`

[Weaviate](https://weaviate.io/) is an open-source vector database that is supported in DocArray.

### `Qdrant`

[Qdrant](https://qdrant.tech/) is an open-source vector database that is supported in DocArray.
