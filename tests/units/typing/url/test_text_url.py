import os
import urllib

import pytest
from pydantic import parse_obj_as, schema_json_of

from docarray.base_doc.io.json import orjson_dumps
from docarray.typing import TextUrl
from tests import TOYDATA_DIR

REMOTE_TEXT_FILE = 'https://www.gutenberg.org/files/1065/1065-0.txt'
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_TEXT_FILES = [
    str(TOYDATA_DIR / 'penal_colony.txt'),
    str(TOYDATA_DIR / 'test.md'),
    str(TOYDATA_DIR / 'test.html'),
    str(TOYDATA_DIR / 'test.css'),
    str(TOYDATA_DIR / 'test.csv'),
    str(TOYDATA_DIR / 'test.log'),
]
LOCAL_TEXT_FILES_AND_BEGINNING = [
    (str(TOYDATA_DIR / 'penal_colony.txt'), '“It’s a peculiar apparatus,”'),
    (str(TOYDATA_DIR / 'test.md'), "# Hello"),
    (str(TOYDATA_DIR / 'test.html'), "<html>"),
    (str(TOYDATA_DIR / 'test.css'), 'body {'),
    (str(TOYDATA_DIR / 'test.csv'), "John,Doe"),
    (str(TOYDATA_DIR / 'test.log'), "2022-11-25 12:34:56 INFO: Program started"),
]


@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize(
    'url,expected_beginning',
    [(REMOTE_TEXT_FILE, 'The Project Gutenberg'), *LOCAL_TEXT_FILES_AND_BEGINNING],
)
def test_load(url, expected_beginning):
    uri = parse_obj_as(TextUrl, url)

    txt = uri.load()
    assert expected_beginning in txt


@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize('url', [REMOTE_TEXT_FILE, *LOCAL_TEXT_FILES])
def test_load_to_bytes(url):
    uri = parse_obj_as(TextUrl, url)

    txt_bytes = uri.load_bytes()
    assert isinstance(txt_bytes, bytes)


@pytest.mark.proto
@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize('url', [REMOTE_TEXT_FILE])
def test_proto_text_url(url):
    uri = parse_obj_as(TextUrl, url)

    proto = uri._to_node_protobuf()
    assert 'text_url' in str(proto)


@pytest.mark.internet
def test_load_timeout():
    url = parse_obj_as(TextUrl, REMOTE_TEXT_FILE)
    with pytest.raises(urllib.error.URLError):
        _ = url.load(timeout=0.001)
    with pytest.raises(urllib.error.URLError):
        _ = url.load_bytes(timeout=0.001)


def test_json_schema():
    schema_json_of(TextUrl)


@pytest.mark.internet
def test_dump_json():
    url = parse_obj_as(TextUrl, REMOTE_TEXT_FILE)
    orjson_dumps(url)


@pytest.mark.parametrize(
    'path_to_file',
    [REMOTE_TEXT_FILE, *LOCAL_TEXT_FILES],
)
def test_validation(path_to_file):
    url = parse_obj_as(TextUrl, path_to_file)
    assert isinstance(url, TextUrl)
    assert isinstance(url, str)


@pytest.mark.parametrize(
    'file_type, file_source',
    [
        *[('text', file) for file in LOCAL_TEXT_FILES],
        ('text', REMOTE_TEXT_FILE),
        ('audio', os.path.join(TOYDATA_DIR, 'hello.aac')),
        ('audio', os.path.join(TOYDATA_DIR, 'hello.mp3')),
        ('audio', os.path.join(TOYDATA_DIR, 'hello.ogg')),
        ('image', os.path.join(TOYDATA_DIR, 'test.png')),
        ('video', os.path.join(TOYDATA_DIR, 'mov_bbb.mp4')),
        ('application', os.path.join(TOYDATA_DIR, 'test.glb')),
    ],
)
def test_file_validation(file_type, file_source):
    if file_type != TextUrl.mime_type():
        with pytest.raises(ValueError):
            parse_obj_as(TextUrl, file_source)
    else:
        parse_obj_as(TextUrl, file_source)
