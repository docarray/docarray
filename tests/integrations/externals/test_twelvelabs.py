# Licensed to the LF AI & Data foundation under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from types import SimpleNamespace

import numpy as np
import pytest

from docarray import DocList
from docarray.documents import TextDoc, VideoDoc

twelvelabs = pytest.importorskip("twelvelabs")

from docarray.utils.twelvelabs import TwelveLabsEmbedder  # noqa: E402

# A publicly reachable HD clip. Marengo requires at least 360p, so the small
# toydata fixture (320x176) cannot be used here.
PUBLIC_VIDEO = (
    'https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/'
    'Big_Buck_Bunny_720_10s_1MB.mp4'
)


class _FakeClient:
    """Minimal stand-in for ``twelvelabs.TwelveLabs`` for no-network tests."""

    def __init__(self, dim: int = 512):
        self._dim = dim

        text_resp = SimpleNamespace(
            text_embedding=SimpleNamespace(
                segments=[SimpleNamespace(float_=[0.1] * dim)]
            )
        )
        retrieve_resp = SimpleNamespace(
            video_embedding=SimpleNamespace(
                segments=[
                    SimpleNamespace(float_=[1.0] * dim),
                    SimpleNamespace(float_=[3.0] * dim),
                ]
            )
        )
        tasks = SimpleNamespace(
            create=lambda **kw: SimpleNamespace(id='task-123'),
            wait_for_done=lambda task_id, **kw: None,
            retrieve=lambda task_id, **kw: retrieve_resp,
        )
        self.embed = SimpleNamespace(
            create=lambda **kw: text_resp,
            tasks=tasks,
        )


def test_embed_text_no_network():
    emb = TwelveLabsEmbedder(client=_FakeClient())
    vec = emb.embed_text('a cat playing piano')
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (512,)
    assert vec.dtype == np.float32


def test_embed_query_accepts_textdoc():
    emb = TwelveLabsEmbedder(client=_FakeClient())
    vec = emb.embed_query(TextDoc(text='hello'))
    assert vec.shape == (512,)


def test_embed_video_url_averages_segments_no_network():
    emb = TwelveLabsEmbedder(client=_FakeClient())
    vec = emb.embed_video_url('http://example.com/clip.mp4')
    # mean of the two fake segments (1.0 and 3.0) is 2.0
    assert vec.shape == (512,)
    np.testing.assert_allclose(vec, np.full(512, 2.0, dtype=np.float32))


def test_embed_docs_in_place_no_network():
    docs = DocList[VideoDoc](
        [
            VideoDoc(url='http://example.com/a.mp4'),
            VideoDoc(url='http://example.com/b.mp4'),
        ]
    )
    emb = TwelveLabsEmbedder(client=_FakeClient())
    emb.embed_docs(docs)
    for d in docs:
        assert d.embedding is not None
        assert np.asarray(d.embedding).shape == (512,)


def test_embed_docs_requires_url():
    emb = TwelveLabsEmbedder(client=_FakeClient())
    with pytest.raises(ValueError):
        emb.embed_docs(VideoDoc())


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv('TWELVELABS_API_KEY', raising=False)
    with pytest.raises(ValueError):
        TwelveLabsEmbedder(api_key=None)


@pytest.mark.skipif(
    not os.environ.get('TWELVELABS_API_KEY'),
    reason='TWELVELABS_API_KEY not set',
)
def test_embed_text_live():
    emb = TwelveLabsEmbedder()
    vec = emb.embed_text('a dog catching a frisbee')
    assert vec.shape == (512,)
    assert np.linalg.norm(vec) > 0


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get('TWELVELABS_API_KEY'),
    reason='TWELVELABS_API_KEY not set',
)
def test_embed_video_live():
    emb = TwelveLabsEmbedder()
    docs = DocList[VideoDoc]([VideoDoc(url=PUBLIC_VIDEO)])
    emb.embed_docs(docs)
    assert docs[0].embedding is not None
    assert np.asarray(docs[0].embedding).shape == (512,)
