__all__ = ['TwelveLabsEmbedder']

import os
from typing import TYPE_CHECKING, List, Optional, Union, overload

import numpy as np

from docarray.documents import TextDoc, VideoDoc
from docarray.utils._internal.misc import import_library

if TYPE_CHECKING:
    from twelvelabs import TwelveLabs


#: Default Marengo embedding model. Marengo produces 512-dimensional
#: multimodal embeddings that live in a single space shared by text, image,
#: audio and video, so a text query embedding can be compared directly against
#: a video embedding.
DEFAULT_MODEL = 'marengo3.0'


class TwelveLabsEmbedder:
    """
    Embed DocArray documents with [TwelveLabs](https://twelvelabs.io) Marengo
    multimodal embeddings.

    Marengo maps text and video into the **same** 512-dimensional vector space,
    which makes it a natural fit for populating the `embedding` field of a
    [`VideoDoc`][docarray.documents.VideoDoc] (or a plain text query) and then
    running cross-modal search with DocArray's
    [`find`][docarray.utils.find.find] / document indexes.

    This is an **opt-in** helper: it is only imported on demand and requires the
    `twelvelabs` extra (`pip install "docarray[twelvelabs]"`) plus a TwelveLabs
    API key.

    ```python
    from docarray import DocList
    from docarray.documents import VideoDoc
    from docarray.utils.twelvelabs import TwelveLabsEmbedder

    embedder = TwelveLabsEmbedder()  # reads TWELVELABS_API_KEY from the env

    # embed a query (synchronous, fast)
    q = embedder.embed_text('a dog catching a frisbee')

    # embed videos (asynchronous on TwelveLabs' side; this blocks until done)
    docs = DocList[VideoDoc](
        [VideoDoc(url='https://example.com/clip.mp4')]
    )
    embedder.embed_docs(docs)
    assert docs[0].embedding is not None
    ```

    :param api_key: TwelveLabs API key. Falls back to the ``TWELVELABS_API_KEY``
        environment variable when not given.
    :param model_name: Marengo model to use. Defaults to ``'marengo3.0'``.
    :param client: An already-configured ``twelvelabs.TwelveLabs`` client to
        reuse instead of creating a new one.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_MODEL,
        client: Optional['TwelveLabs'] = None,
    ) -> None:
        self.model_name = model_name
        if client is not None:
            self._client = client
        else:
            twelvelabs = import_library('twelvelabs')
            key = api_key or os.environ.get('TWELVELABS_API_KEY')
            if not key:
                raise ValueError(
                    'A TwelveLabs API key is required. Pass `api_key=...` or set '
                    'the `TWELVELABS_API_KEY` environment variable. You can get a '
                    'free key at https://twelvelabs.io .'
                )
            self._client = twelvelabs.TwelveLabs(api_key=key)

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a text string into a 512-dimensional Marengo vector.

        :param text: the text to embed (e.g. a search query).
        :return: a ``numpy`` array of shape ``(512,)``.
        """
        resp = self._client.embed.create(model_name=self.model_name, text=text)
        segment = resp.text_embedding.segments[0]
        return np.asarray(segment.float_, dtype=np.float32)

    def embed_video_url(
        self,
        url: str,
        *,
        clip_length: Optional[float] = None,
        sleep_interval: float = 5.0,
    ) -> np.ndarray:
        """
        Embed a (publicly reachable) video URL into a single 512-dimensional
        Marengo vector.

        Video embedding is asynchronous on TwelveLabs' side: this method starts
        an embedding task, blocks until it finishes, and returns the mean of the
        per-segment ``'video'``-scope embeddings so that the whole clip is
        represented by one vector that lives in the same space as
        :meth:`embed_text`.

        :param url: a publicly reachable URL that TwelveLabs can fetch.
        :param clip_length: optional fixed clip length (seconds) for segmentation.
        :param sleep_interval: polling interval (seconds) while waiting.
        :return: a ``numpy`` array of shape ``(512,)``.
        """
        create_kwargs: dict = {'model_name': self.model_name, 'video_url': url}
        if clip_length is not None:
            create_kwargs['video_clip_length'] = clip_length
        task = self._client.embed.tasks.create(**create_kwargs)
        self._client.embed.tasks.wait_for_done(task.id, sleep_interval=sleep_interval)
        result = self._client.embed.tasks.retrieve(task.id)
        segments = result.video_embedding.segments
        vectors = [np.asarray(s.float_, dtype=np.float32) for s in segments]
        if not vectors:
            raise RuntimeError(
                f'TwelveLabs returned no embedding segments for task {task.id!r}.'
            )
        return np.mean(vectors, axis=0)

    @overload
    def embed_docs(self, docs: VideoDoc) -> VideoDoc:
        ...

    @overload
    def embed_docs(self, docs: List[VideoDoc]) -> List[VideoDoc]:
        ...

    def embed_docs(
        self,
        docs: Union[VideoDoc, List[VideoDoc]],
        **kwargs,
    ):
        """
        Populate the ``embedding`` field of one or more
        [`VideoDoc`][docarray.documents.VideoDoc] in place using their ``url``.

        :param docs: a single ``VideoDoc`` or an iterable of them. Each must have
            its ``url`` set.
        :param kwargs: forwarded to :meth:`embed_video_url`
            (e.g. ``clip_length``, ``sleep_interval``).
        :return: the same ``docs`` object, embedded in place.
        """
        single = isinstance(docs, VideoDoc)
        items = [docs] if single else list(docs)
        for doc in items:
            if doc.url is None:
                raise ValueError(
                    'VideoDoc.url must be set to embed it with TwelveLabs.'
                )
            doc.embedding = self.embed_video_url(str(doc.url), **kwargs)
        return docs

    def embed_query(self, query: Union[str, TextDoc]) -> np.ndarray:
        """
        Convenience wrapper to embed a text query, accepting either a raw string
        or a [`TextDoc`][docarray.documents.TextDoc].

        :param query: the query text or ``TextDoc``.
        :return: a ``numpy`` array of shape ``(512,)``.
        """
        text = query.text if isinstance(query, TextDoc) else query
        if text is None:
            raise ValueError('Cannot embed a TextDoc with no `text`.')
        return self.embed_text(text)
