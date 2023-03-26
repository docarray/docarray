import os
from typing import Optional

import numpy as np
import pytest
import torch
from pydantic import parse_obj_as

from docarray import BaseDocument
from docarray.documents import AudioDoc
from docarray.typing import AudioUrl
from docarray.typing.tensor.audio import AudioNdArray, AudioTorchTensor
from docarray.utils.misc import is_tf_available
from tests import TOYDATA_DIR

tf_available = is_tf_available()
if tf_available:
    import tensorflow as tf
    import tensorflow._api.v2.experimental.numpy as tnp

    from docarray.typing.tensor import TensorFlowTensor
    from docarray.typing.tensor.audio import AudioTensorFlowTensor

AUDIODATA_DIR = TOYDATA_DIR / 'audio-data'

LOCAL_AUDIO_FILES = [
    str(AUDIODATA_DIR / 'hello.wav'),
    str(AUDIODATA_DIR / 'olleh.wav'),
    str(AUDIODATA_DIR / 'hello.mp3'),
    str(AUDIODATA_DIR / 'hello.flac'),
    str(AUDIODATA_DIR / 'hello.ogg'),
    str(AUDIODATA_DIR / 'hello.wma'),
    str(AUDIODATA_DIR / 'hello.aac'),
    str(AUDIODATA_DIR / 'hello'),
]

LOCAL_AUDIO_FILES_AND_FORMAT = [
    (str(AUDIODATA_DIR / 'hello.wav'), 'wav'),
    (str(AUDIODATA_DIR / 'olleh.wav'), 'wav'),
    (str(AUDIODATA_DIR / 'hello.mp3'), 'mp3'),
    (str(AUDIODATA_DIR / 'hello.flac'), 'flac'),
    (str(AUDIODATA_DIR / 'hello.ogg'), 'ogg'),
    (str(AUDIODATA_DIR / 'hello.wma'), 'asf'),
    (str(AUDIODATA_DIR / 'hello.aac'), 'adts'),
    (str(AUDIODATA_DIR / 'hello'), 'wav'),
]

NON_AUDIO_FILES = [
    str(TOYDATA_DIR / 'captions.csv'),
    str(TOYDATA_DIR / 'cube.ply'),
    str(TOYDATA_DIR / 'test.glb'),
    str(TOYDATA_DIR / 'test.png'),
    'illegal',
    'https://www.github.com',
]


@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize('file_url', LOCAL_AUDIO_FILES)
def test_audio(file_url):
    audio = AudioDoc(url=file_url)
    audio.tensor, _ = audio.url.load()
    assert isinstance(audio.tensor, np.ndarray)


@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize('file_url', NON_AUDIO_FILES)
def test_non_audio(file_url):
    with pytest.raises(Exception):
        audio = AudioDoc(url=file_url)
        _, _ = audio.url.load()


@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize('file_url, format', LOCAL_AUDIO_FILES_AND_FORMAT)
def test_save_audio_ndarray(file_url, format, tmpdir):
    filename = os.path.basename(file_url)
    tmp_file = str(tmpdir / filename)

    audio = AudioDoc(url=file_url)
    audio.tensor, audio.frame_rate = audio.url.load()
    assert isinstance(audio.tensor, np.ndarray)
    assert isinstance(audio.tensor, AudioNdArray)

    audio.tensor.save(tmp_file, format=format, frame_rate=audio.frame_rate)
    assert os.path.isfile(tmp_file)

    audio_from_file = AudioDoc(url=tmp_file)
    audio_from_file.tensor, _ = audio_from_file.url.load()
    if format in ['wav', 'flac']:
        # lossless formats (wav, flac) can be loaded back exactly
        assert np.allclose(audio.tensor, audio_from_file.tensor)
    elif format in ['mp3']:
        # lossy formats, we can only check the shape
        assert audio.tensor.shape == audio_from_file.tensor.shape
    else:
        # encoding to other formats may change the shape, only check file exists
        pass


@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize('file_url, format', LOCAL_AUDIO_FILES_AND_FORMAT)
def test_save_audio_torch_tensor(file_url, format, tmpdir):
    tmp_file = str(tmpdir / 'tmp.wav')

    audio = AudioDoc(url=file_url)
    tensor, frame_rate = audio.url.load()

    audio.tensor = parse_obj_as(AudioTorchTensor, torch.from_numpy(tensor))
    assert isinstance(audio.tensor, torch.Tensor)
    assert isinstance(audio.tensor, AudioTorchTensor)

    audio.tensor.save(tmp_file, format=format, frame_rate=frame_rate)

    assert os.path.isfile(tmp_file)

    audio_from_file = AudioDoc(url=tmp_file)
    tensor, _ = audio_from_file.url.load()
    audio_from_file.tensor = parse_obj_as(AudioTorchTensor, torch.from_numpy(tensor))
    if format in ['wav', 'flac']:
        # lossless formats (wav, flac) can be loaded back exactly
        assert np.allclose(audio.tensor, audio_from_file.tensor)
    elif format in ['mp3']:
        # lossy formats, we can only check the shape
        assert audio.tensor.shape == audio_from_file.tensor.shape
    else:
        # encoding to other formats may change the shape, only check file exists
        pass


@pytest.mark.tensorflow
@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize('file_url, format', LOCAL_AUDIO_FILES_AND_FORMAT)
def test_save_audio_tensorflow(file_url, format, tmpdir):
    tmp_file = str(tmpdir / 'tmp.wav')

    audio = AudioDoc(url=file_url)
    tensor, frame_rate = audio.url.load()
    audio.tensor = AudioTensorFlowTensor(tensor=tf.constant(tensor))
    assert isinstance(audio.tensor, TensorFlowTensor)
    assert isinstance(audio.tensor, AudioTensorFlowTensor)
    assert isinstance(audio.tensor.tensor, tf.Tensor)

    audio.tensor.save(tmp_file, format=format, frame_rate=frame_rate)
    assert os.path.isfile(tmp_file)

    audio_from_file = AudioDoc(url=tmp_file)
    tensor, _ = audio_from_file.url.load()
    audio_from_file.tensor = AudioTensorFlowTensor(tensor=tf.constant(tensor))
    if format in ['wav', 'flac']:
        # lossless formats (wav, flac) can be loaded back exactly
        assert tnp.allclose(audio.tensor.tensor, audio_from_file.tensor.tensor)
    elif format in ['mp3']:
        # lossy formats, we can only check the shape
        assert audio.tensor.tensor.shape == audio_from_file.tensor.tensor.shape
    else:
        # encoding to other formats may change the shape, only check file exists
        pass


@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize(
    'file_url',
    LOCAL_AUDIO_FILES,
)
def test_extend_audio(file_url):
    class MyAudio(AudioDoc):
        title: str
        tensor: Optional[AudioNdArray]

    my_audio = MyAudio(title='my extended audio', url=file_url)
    tensor, _ = my_audio.url.load()
    my_audio.tensor = parse_obj_as(AudioNdArray, tensor)

    assert isinstance(my_audio.tensor, AudioNdArray)
    assert isinstance(my_audio.url, AudioUrl)


def test_audio_np():
    audio = parse_obj_as(AudioDoc, np.zeros((10, 10, 3)))
    assert (audio.tensor == np.zeros((10, 10, 3))).all()


def test_audio_torch():
    audio = parse_obj_as(AudioDoc, torch.zeros(10, 10, 3))
    assert (audio.tensor == torch.zeros(10, 10, 3)).all()


@pytest.mark.tensorflow
def test_audio_tensorflow():
    audio = parse_obj_as(AudioDoc, tf.zeros((10, 10, 3)))
    assert tnp.allclose(audio.tensor.tensor, tf.zeros((10, 10, 3)))


def test_audio_bytes():
    audio = parse_obj_as(AudioDoc, torch.zeros(10, 10, 3))
    audio.bytes_ = audio.tensor.to_bytes()


def test_audio_shortcut_doc():
    class MyDoc(BaseDocument):
        audio: AudioDoc
        audio2: AudioDoc
        audio3: AudioDoc

    doc = MyDoc(
        audio='http://myurl.wav',
        audio2=np.zeros((10, 10, 3)),
        audio3=torch.zeros(10, 10, 3),
    )
    assert doc.audio.url == 'http://myurl.wav'
    assert (doc.audio2.tensor == np.zeros((10, 10, 3))).all()
    assert (doc.audio3.tensor == torch.zeros(10, 10, 3)).all()
