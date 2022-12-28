import os
from typing import Optional

import numpy as np
import pytest
import torch
from pydantic import parse_obj_as

from docarray import Audio
from docarray.typing import AudioUrl
from docarray.typing.tensor.audio import AudioNdArray, AudioTorchTensor
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
def test_save_audio_ndarray(file_url, tmpdir):
    tmp_file = str(tmpdir / 'tmp.wav')

    audio = Audio(url=file_url)
    audio.tensor = parse_obj_as(AudioNdArray, audio.url.load())
    assert isinstance(audio.tensor, np.ndarray)

    audio.tensor.save_to_wav_file(tmp_file)
    assert os.path.isfile(tmp_file)

    audio_from_file = Audio(url=tmp_file)
    audio_from_file.tensor = audio_from_file.url.load()
    assert (audio.tensor == audio_from_file.tensor).all()


@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize('file_url', [*LOCAL_AUDIO_FILES, REMOTE_AUDIO_FILE])
def test_save_audio_torch_tensor(file_url, tmpdir):
    tmp_file = str(tmpdir / 'tmp.wav')

    audio = Audio(url=file_url)
    audio.tensor = parse_obj_as(AudioTorchTensor, torch.from_numpy(audio.url.load()))
    assert isinstance(audio.tensor, torch.Tensor)
    assert isinstance(audio.tensor, AudioTorchTensor)

    audio.tensor.save_to_wav_file(tmp_file)
    assert os.path.isfile(tmp_file)

    audio_from_tmp = Audio(url=tmp_file)
    audio_from_tmp.tensor = parse_obj_as(
        AudioTorchTensor, torch.from_numpy(audio_from_tmp.url.load())
    )
    assert (audio.tensor == audio_from_tmp.tensor).all()


@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize(
    'file_url',
    [*LOCAL_AUDIO_FILES, REMOTE_AUDIO_FILE],
)
def test_extend_audio(file_url):
    class MyAudio(Audio):
        title: str
        tensor: Optional[AudioNdArray]

    my_audio = MyAudio(title='my extended audio', url=file_url)
    my_audio.tensor = parse_obj_as(AudioNdArray, my_audio.url.load())

    assert isinstance(my_audio.tensor, AudioNdArray)
    assert isinstance(my_audio.url, AudioUrl)
