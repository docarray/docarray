# Creating an Audio to Text App with Jina and DocArray V2

This is how you can build an Audio to Text app using both Jina and DocarrayV2

We will use: 

* DocarrayV2: Helps us to load and preprocess multimodal data such as image, text and audio in our case
* Jina: Helps us serve the model quickly and create a client

First let's install requirements

## 💾 Installation

```bash
pip install transformers
pip install openai-whisper
pip install jina
```

Now let's import necessary libraries


```python
import whisper
from jina import Executor, requests, Deployment
from docarray import BaseDoc, DocList
from docarray.typing import AudioUrl
```

Now we need to create the schema of our input and output documents. Since our input is an audio
our input schema should contain an AudioUrl like the following

```python
class AudioURL(BaseDoc):
    audio: AudioUrl
```

As for the output schema we would like to receive the transcribed text so we use the following:

```python
class Response(BaseDoc):
    text: str
```

Now it's time we create our model, we wrap our model into Jina Executor, this allows us to serve to model
later on and expose its endpoint /transcribe

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

Now we can leverage Deployment object provided by Jina to use this executor
then we send a request to transcribe endpoint. Here we are using an audio file previously recorded
that says, "A Man reading a book" saved under resources/audio.mp3 but feel free to use your own audio.

```python
with Deployment(
    uses=WhisperExecutor, uses_with={'device': "cpu"}, port=12349, timeout_ready=-1
) as d:
    docs = d.post(
        on='/transcribe',
        inputs=[AudioURL(audio='resources/audio.mp3')],
        return_type=DocList[Response],
    )
    print(docs[0].text)
```

And we get the transcribed result!