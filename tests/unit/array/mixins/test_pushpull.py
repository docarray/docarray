import cgi
import os
from io import BytesIO

import pytest
import requests
from hubble import Client

from docarray import DocumentArray, Document, dataclass
from docarray.helper import random_identity
from docarray.typing import Image, Text
from tests import random_docs


class PushMockResponse:
    def __init__(self, status_code: int = 200):
        self.status_code = status_code
        self.headers = {'Content-length': 1}
        self.ok = status_code == 200

    def json(self):
        return {'code': self.status_code, 'data': []}

    def raise_for_status(self):
        raise Exception


class PullMockResponse:
    def __init__(self, status_code: int = 200):
        self.status_code = status_code
        self.headers = {'Content-length': 1}
        self.ok = True

    def json(self):
        return {
            'code': self.status_code,
            'data': {'download': 'http://test_download.com/test.tar.gz'},
        }


class DownloadMockResponse:
    def __init__(self, status_code: int = 200):
        self.status_code = status_code
        self.headers = {'Content-length': 1}

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        for _ in range(10):
            yield b'' * chunk_size

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def _mock_post(mock, monkeypatch, status_code=requests.codes.ok):
    def _mocker(url, data, headers=None):
        mock(url=url, data=data, headers=headers)
        return PushMockResponse(status_code=status_code)

    monkeypatch.setattr(requests, 'post', _mocker)


def _mock_get_user_info(mock, monkeypatch):
    def _get_user_info(obj):
        return {}

    monkeypatch.setattr(Client, 'get_user_info', _get_user_info)


def _mock_get(mock, monkeypatch, status_code=requests.codes.ok):
    def _mocker(url, stream=False, headers=None):
        mock(url=url, stream=stream, headers=headers)
        if stream:
            return DownloadMockResponse(status_code=status_code)
        return PullMockResponse(status_code=status_code)

    monkeypatch.setattr(requests, 'get', _mocker)


def test_push(mocker, monkeypatch):
    mock = mocker.Mock()
    _mock_post(mock, monkeypatch)

    docs = random_docs(2)
    name = random_identity()
    docs.push(name)

    assert mock.call_count == 1


@pytest.mark.parametrize('public', [True, False])
def test_push_with_public(mocker, monkeypatch, public):
    mock = mocker.Mock()
    _mock_post(mock, monkeypatch)

    docs = random_docs(2)
    name = random_identity()
    docs.push(name, public=public)

    _, mock_kwargs = mock.call_args_list[0]

    c_type, c_data = cgi.parse_header(mock_kwargs['headers']['Content-Type'])
    assert c_type == 'multipart/form-data'

    form_data = cgi.parse_multipart(
        BytesIO(b''.join(mock_kwargs['data'])),
        {'boundary': c_data['boundary'].encode()},
    )

    assert form_data['public'] == [str(public)]


@pytest.mark.parametrize('da_name', ['test_name', 'username/test_name'])
def test_pull(mocker, monkeypatch, da_name):
    mock = mocker.Mock()
    _mock_get(mock, monkeypatch)

    DocumentArray.pull(name=da_name)

    assert mock.call_count == 2  # 1 for pull, 1 for download

    _, pull_kwargs = mock.call_args_list[0]
    _, download_kwargs = mock.call_args_list[1]

    assert pull_kwargs['stream'] is False

    assert download_kwargs['stream'] is True
    assert download_kwargs['url'] == 'http://test_download.com/test.tar.gz'


def test_push_fail(mocker, monkeypatch):
    mock = mocker.Mock()
    _mock_post(mock, monkeypatch, status_code=requests.codes.forbidden)

    docs = random_docs(2)
    with pytest.raises(Exception):
        docs.push('test_name')

    assert mock.call_count == 1


@pytest.fixture()
def set_hubble_registry():
    os.environ['JINA_HUBBLE_REGISTRY'] = 'http://localhost:8080'
    yield
    del os.environ['JINA_HUBBLE_REGISTRY']


def test_api_url_change(mocker, monkeypatch, set_hubble_registry):

    mock = mocker.Mock()
    _mock_post(mock, monkeypatch)
    _mock_get(mock, monkeypatch)
    _mock_get_user_info(mock, monkeypatch)

    docs = random_docs(2)
    name = random_identity()
    docs.push(name)
    docs.pull(name)

    assert (
        mock.call_count >= 3
    )  # at least 1 for push, 1 for pull, 1 for download + extra for auth

    _, push_kwargs = mock.call_args_list[0]
    _, pull_kwargs = mock.call_args_list[1]

    test_api_url = 'http://localhost:8080'
    assert push_kwargs['url'].startswith(test_api_url)
    assert pull_kwargs['url'].startswith(test_api_url)


@dataclass
class MyDocument:
    image: Image
    paragraph: Text


@pytest.mark.parametrize(
    'da',
    [
        DocumentArray(),
        DocumentArray.empty(10),
        DocumentArray.empty(10, storage='annlite', config={'n_dim': 10}),
        DocumentArray(
            [
                Document(
                    MyDocument(
                        image='https://docarray.jina.ai/_images/apple.png',
                        paragraph='hello world',
                    )
                )
                for _ in range(10)
            ],
            config={'n_dim': 256},
            storage='annlite',
            subindex_configs={
                '@.[image]': {'n_dim': 512},
                '@.[paragraph]': {'n_dim': 128},
            },
        ),
    ],
)
def test_get_raw_summary(da: DocumentArray):
    assert da._get_raw_summary()
