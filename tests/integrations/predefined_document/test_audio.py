import os

import numpy as np
import pytest

from docarray import Audio
from tests import TOYDATA_DIR

LOCAL_AUDIO_FILES = [
    str(TOYDATA_DIR / 'hello.wav'),
    str(TOYDATA_DIR / 'olleh.wav'),
]
REMOTE_AUDIO_FILE = 'https://www.kozco.com/tech/piano2.wav'


@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize('file_url', [*LOCAL_AUDIO_FILES, REMOTE_AUDIO_FILE])
def test_audio(file_url):

    audio = Audio(url=file_url)

    audio.tensor = audio.url.load()

    assert isinstance(audio.tensor, np.ndarray)


@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize('file_url', [*LOCAL_AUDIO_FILES, REMOTE_AUDIO_FILE])
def test_save_audio_tensor(file_url):
    tmp_file = str(TOYDATA_DIR / 'tmp.wav')

    audio = Audio(url=file_url)
    audio.tensor = audio.url.load()
    audio.save_audio_tensor_to_file(tmp_file)
    assert os.path.isfile(tmp_file)

    audio_from_file = Audio(url=tmp_file)
    audio_from_file.tensor = audio_from_file.url.load()
    assert (audio.tensor == audio_from_file.tensor).all()

    os.remove(tmp_file)
