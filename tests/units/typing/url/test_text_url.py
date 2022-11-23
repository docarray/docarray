import os
import urllib

import pytest
from pydantic import parse_obj_as, schema_json_of

from docarray.document.io.json import orjson_dumps
from docarray.typing import TextUrl

REMOTE_TXT = 'https://de.wikipedia.org/wiki/Brixen'
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_TXT = os.path.join(CUR_DIR, '..', '..', '..', 'toydata', 'penal_colony.txt')


@pytest.mark.parametrize(
    'url,expected_beginning',
    [(REMOTE_TXT, '<!DOCTYPE html>'), (LOCAL_TXT, '“It’s a peculiar apparatus,”')],
)
def test_load(url, expected_beginning):
    uri = parse_obj_as(TextUrl, url)

    txt = uri.load()
    assert txt.startswith(expected_beginning)


@pytest.mark.parametrize('url', [REMOTE_TXT, LOCAL_TXT])
def test_load_to_bytes(url):
    uri = parse_obj_as(TextUrl, url)

    txt_bytes = uri.load_to_bytes()
    assert isinstance(txt_bytes, bytes)


def test_proto_text_url():

    uri = parse_obj_as(TextUrl, LOCAL_TXT)

    uri._to_node_protobuf()


def test_load_timeout():
    url = parse_obj_as(TextUrl, REMOTE_TXT)
    with pytest.raises(urllib.error.URLError):
        _ = url.load(timeout=0.001)
    with pytest.raises(urllib.error.URLError):
        _ = url.load_to_bytes(timeout=0.001)


def test_json_schema():
    schema_json_of(TextUrl)


def test_dump_json():
    url = parse_obj_as(TextUrl, REMOTE_TXT)
    orjson_dumps(url)
