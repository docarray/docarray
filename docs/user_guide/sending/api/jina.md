# Send over Jina

In this example we'll build an audio-to-text app using [Jina](https://docs.jina.ai/), DocArray and [Whisper](https://openai.com/research/whisper).

We will use: 

* DocArray V2: To load and preprocess multimodal data such as image, text and audio.
* Jina: To serve the model quickly and create a client.

## Install packages

First let's install requirements:

```bash
pip install transformers
pip install openai-whisper
pip install jina
```

## Import libraries

Let's import the necessary libraries:

```python
import whisper
from jina import Executor, requests, Deployment
from docarray import BaseDoc, DocList
from docarray.typing import AudioUrl
```

## Create schemeas

Now we need to create the schema of our input and output documents. Since our input is an audio URL,
our input schema should contain an `AudioUrl`:

```python
class AudioURL(BaseDoc):
    audio: AudioUrl
```

For the output schema we would like to receive the transcribed text:

```python
class Response(BaseDoc):
    text: str
```

## Create Executor

To create our model, we wrap our model into a Jina [Executor](https://docs.jina.ai/concepts/serving/executor/), allowing us to serve the model
later and expose the endpoint `/transcribe`:

```python
class WhisperExecutor(Executor):
    def __init__(self, device: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = whisper.load_model("medium.en", device=device)

    @requests
    def transcribe(self, docs: DocList[AudioURL], **kwargs) -> DocList[Response]:
        response_docs = DocList[Response]()
        for doc in docs:
            transcribed_text = self.model.transcribe(str(doc.audio))['text']
            response_docs.append(Response(text=transcribed_text))

        return response_docs
```

## Deploy Executor and get results

Now we can leverage Jina's [Deployment object](https://docs.jina.ai/concepts/orchestration/deployment/) to deploy this Executor, then send a request to the `/transcribe` endpoint. 

Here we are using an audio file that says, "A man reading a book", saved as `resources/audio.mp3`:

```python
dep = Deployment(
    uses=WhisperExecutor, uses_with={'device': "cpu"}, port=12349, timeout_ready=-1
)

with dep:
    docs = d.post(
        on='/transcribe',
        inputs=[AudioURL(audio='resources/audio.mp3')],
        return_type=DocList[Response],
    )

print(docs[0].text)
```

And we get the transcribed result:

```text
A man reading a book
```
