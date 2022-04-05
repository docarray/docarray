import json
import os
import pytest
import requests

from docarray import DocumentArray
from docarray.array.mixins.io.pushpull import JINA_CLOUD_CONFIG

from tests import random_docs


class PushMockResponse:
    def __init__(self, status_code: int = 200):
        self.status_code = status_code
        self.headers = {'Content-length': 1}

    def json(self):
        return {'code': self.status_code}


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
    docs.push(name='test_name')

    assert mock.call_count == 1


def test_pull(mocker, monkeypatch):
    mock = mocker.Mock()
    _mock_get(mock, monkeypatch)

    DocumentArray.pull(name='test_name')

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
    with pytest.raises(RuntimeError) as exc_info:
        docs.push('test_name')

    assert exc_info.match('Failed to push DocumentArray to Jina Cloud')
    assert exc_info.match('Status code: 403')
    assert mock.call_count == 1


def test_api_url_change(mocker, monkeypatch):
    from docarray.array.mixins.io.pushpull import _get_cloud_api

    _get_cloud_api.cache_clear()
    test_api_url = 'http://localhost:8080'
    os.environ['JINA_HUBBLE_REGISTRY'] = test_api_url

    mock = mocker.Mock()
    _mock_post(mock, monkeypatch)
    _mock_get(mock, monkeypatch)

    docs = random_docs(2)
    docs.push(name='test_name')
    docs.pull(name='test_name')

    del os.environ['JINA_HUBBLE_REGISTRY']
    _get_cloud_api.cache_clear()

    assert mock.call_count == 3  # 1 for push, 1 for pull, 1 for download

    _, push_kwargs = mock.call_args_list[0]
    _, pull_kwargs = mock.call_args_list[1]

    assert push_kwargs['url'].startswith(test_api_url)
    assert pull_kwargs['url'].startswith(test_api_url)


def test_api_authorization_header(mocker, monkeypatch, tmpdir):
    from docarray.array.mixins.io.pushpull import _get_hub_config

    _get_hub_config.cache_clear()
    os.environ['JINA_HUB_ROOT'] = str(tmpdir)

    token = 'test-auth-token'
    with open(tmpdir / JINA_CLOUD_CONFIG, 'w') as f:
        json.dump({'auth_token': token}, f)

    mock = mocker.Mock()
    _mock_post(mock, monkeypatch)
    _mock_get(mock, monkeypatch)

    docs = random_docs(2)
    docs.push(name='test_name')
    DocumentArray.pull(name='test_name')

    del os.environ['JINA_HUB_ROOT']
    _get_hub_config.cache_clear()

    assert mock.call_count == 3  # 1 for push, 1 for pull, 1 for download

    _, push_kwargs = mock.call_args_list[0]
    _, pull_kwargs = mock.call_args_list[1]

    assert push_kwargs['headers'].get('Authorization') == f'token {token}'
    assert pull_kwargs['headers'].get('Authorization') == f'token {token}'
