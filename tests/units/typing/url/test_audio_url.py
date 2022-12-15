import numpy as np
import pytest
from pydantic.tools import parse_obj_as, schema_json_of

from docarray.document.io.json import orjson_dumps
from docarray.typing import AudioUrl
from tests import TOYDATA_DIR

AUDIO_FILES = [
    str(TOYDATA_DIR / 'hello.wav'),
    str(TOYDATA_DIR / 'olleh.wav'),
]
REMOTE_AUDIO_FILE = 'https://www.kozco.com/tech/piano2.wav'


@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize(
    'file_url',
    [*AUDIO_FILES, REMOTE_AUDIO_FILE],
)
def test_audio_url(file_url):
    uri = parse_obj_as(AudioUrl, file_url)
    tensor = uri.load()
    assert isinstance(tensor, np.ndarray)


@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize(
    'file_url',
    [*AUDIO_FILES, REMOTE_AUDIO_FILE],
)
def test_load(file_url):
    url = parse_obj_as(AudioUrl, file_url)
    tensor = url.load()
    assert isinstance(tensor, np.ndarray)


def test_json_schema():
    schema_json_of(AudioUrl)


def test_dump_json():
    url = parse_obj_as(AudioUrl, REMOTE_AUDIO_FILE)
    orjson_dumps(url)


@pytest.mark.parametrize(
    'file_format,path_to_file',
    [
        *[('wav', file) for file in AUDIO_FILES],
        ('wav', REMOTE_AUDIO_FILE),
        ('illegal', 'illegal'),
        ('illegal', 'https://www.google.com'),
        ('illegal', 'my/local/text/file.txt'),
        ('illegal', 'my/local/text/file.png'),
    ],
)
def test_validation(file_format, path_to_file):
    if file_format == 'illegal':
        with pytest.raises(ValueError, match='AudioUrl'):
            parse_obj_as(AudioUrl, path_to_file)
    else:
        url = parse_obj_as(AudioUrl, path_to_file)
        assert isinstance(url, AudioUrl)
        assert isinstance(url, str)


@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize(
    'file_url',
    [*AUDIO_FILES, REMOTE_AUDIO_FILE],
)
def test_proto_audio_url(file_url):
    uri = parse_obj_as(AudioUrl, file_url)
    uri._to_node_protobuf()
