import os
import pytest
import requests

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
        return []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def test_push(mocker, monkeypatch):
    mock = mocker.Mock()

    def _mock_post(url, data, headers=None):
        mock(url=url, data=data)
        return PushMockResponse(status_code=requests.codes.ok)

    monkeypatch.setattr(requests, 'post', _mock_post)

    docs = random_docs(2)
    docs.push('test_token')

    assert mock.call_count == 1


def test_pull(mocker, monkeypatch):
    mock = mocker.Mock()

    def _mock_get(url, stream=False, headers=None):
        mock(url=url, stream=stream)
        if stream:
            return DownloadMockResponse(status_code=requests.codes.ok)
        return PullMockResponse(status_code=requests.codes.ok)

    monkeypatch.setattr(requests, 'get', _mock_get)

    docs = random_docs(2)
    docs.pull('test_token')

    assert mock.call_count == 2  # 1 for pull, 1 for download

    _, pull_kwargs = mock.call_args_list[0]
    _, download_kwargs = mock.call_args_list[1]

    assert pull_kwargs['stream'] is False

    assert download_kwargs['stream'] is True
    assert download_kwargs['url'] == 'http://test_download.com/test.tar.gz'


def test_push_fail(mocker, monkeypatch):
    mock = mocker.Mock()

    def _mock_post(url, data, headers=None):
        mock(url=url, data=data)
        return PushMockResponse(status_code=requests.codes.forbidden)

    monkeypatch.setattr(requests, 'post', _mock_post)

    docs = random_docs(2)
    with pytest.raises(RuntimeError) as exc_info:
        docs.push('test_token')

    assert exc_info.match('Failed to push DocumentArray to Jina Cloud')
    assert exc_info.match('Status code: 403')
    assert mock.call_count == 1


def test_api_url_change(mocker, monkeypatch):
    from docarray.array.mixins.io.pushpull import _get_cloud_api

    _get_cloud_api.cache_clear()
    test_api_url = 'http://localhost:8080'
    os.environ['JINA_HUBBLE_REGISTRY'] = test_api_url

    mock = mocker.Mock()

    def _mock_post(url, data, headers=None):
        mock(url=url, data=data)
        return PushMockResponse(status_code=requests.codes.ok)

    def _mock_get(url, stream=False, headers=None):
        mock(url=url, stream=stream)
        if stream:
            return DownloadMockResponse(status_code=requests.codes.ok)
        return PullMockResponse(status_code=requests.codes.ok)

    monkeypatch.setattr(requests, 'post', _mock_post)
    monkeypatch.setattr(requests, 'get', _mock_get)

    docs = random_docs(2)
    docs.push('test_token')
    docs.pull('test_token')

    del os.environ['JINA_HUBBLE_REGISTRY']
    _get_cloud_api.cache_clear()

    assert mock.call_count == 3  # 1 for push, 1 for pull, 1 for download

    _, push_kwargs = mock.call_args_list[0]
    _, pull_kwargs = mock.call_args_list[1]

    assert push_kwargs['url'].startswith(test_api_url)
    assert pull_kwargs['url'].startswith(test_api_url)
