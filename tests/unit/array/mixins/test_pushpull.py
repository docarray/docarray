import cgi
import json
import os
import pytest
import requests
from io import BytesIO

from docarray import DocumentArray
from docarray.array.mixins.io.pushpull import JINA_CLOUD_CONFIG
from docarray.helper import random_identity

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
    with pytest.raises(Exception):
        docs.push('test_name')

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
    name = random_identity()
    docs.push(name)
    docs.pull(name)

    del os.environ['JINA_HUBBLE_REGISTRY']
    _get_cloud_api.cache_clear()

    assert mock.call_count == 3  # 1 for push, 1 for pull, 1 for download

    _, push_kwargs = mock.call_args_list[0]
    _, pull_kwargs = mock.call_args_list[1]

    assert push_kwargs['url'].startswith(test_api_url)
    assert pull_kwargs['url'].startswith(test_api_url)


def test_api_authorization_header_from_config(mocker, monkeypatch, tmpdir):
    from docarray.array.mixins.io.pushpull import _get_hub_config, _get_auth_token

    _get_hub_config.cache_clear()
    _get_auth_token.cache_clear()

    os.environ['JINA_HUB_ROOT'] = str(tmpdir)

    token = 'test-auth-token'
    with open(tmpdir / JINA_CLOUD_CONFIG, 'w') as f:
        json.dump({'auth_token': token}, f)

    mock = mocker.Mock()
    _mock_post(mock, monkeypatch)
    _mock_get(mock, monkeypatch)

    docs = random_docs(2)
    name = random_identity()
    docs.push(name)
    DocumentArray.pull(name)

    del os.environ['JINA_HUB_ROOT']

    _get_hub_config.cache_clear()
    _get_auth_token.cache_clear()

    assert mock.call_count == 3  # 1 for push, 1 for pull, 1 for download

    _, push_kwargs = mock.call_args_list[0]
    _, pull_kwargs = mock.call_args_list[1]

    assert push_kwargs['headers'].get('Authorization') == f'token {token}'
    assert pull_kwargs['headers'].get('Authorization') == f'token {token}'


@pytest.mark.parametrize(
    'set_env_vars', [{'JINA_AUTH_TOKEN': 'test-auth-token'}], indirect=True
)
def test_api_authorization_header_from_env(mocker, monkeypatch, set_env_vars):
    from docarray.array.mixins.io.pushpull import _get_hub_config, _get_auth_token

    _get_hub_config.cache_clear()
    _get_auth_token.cache_clear()

    mock = mocker.Mock()
    _mock_post(mock, monkeypatch)
    _mock_get(mock, monkeypatch)

    docs = random_docs(2)
    name = random_identity()
    docs.push(name)
    DocumentArray.pull(name)

    _get_hub_config.cache_clear()
    _get_auth_token.cache_clear()

    assert mock.call_count == 3  # 1 for push, 1 for pull, 1 for download

    _, push_kwargs = mock.call_args_list[0]
    _, pull_kwargs = mock.call_args_list[1]

    assert push_kwargs['headers'].get('Authorization') == 'token test-auth-token'
    assert pull_kwargs['headers'].get('Authorization') == 'token test-auth-token'


@pytest.mark.parametrize(
    'set_env_vars', [{'JINA_AUTH_TOKEN': 'test-auth-token-env'}], indirect=True
)
def test_api_authorization_header_env_and_config(
    mocker, monkeypatch, tmpdir, set_env_vars
):
    from docarray.array.mixins.io.pushpull import _get_hub_config, _get_auth_token

    _get_hub_config.cache_clear()
    _get_auth_token.cache_clear()

    os.environ['JINA_HUB_ROOT'] = str(tmpdir)

    token = 'test-auth-token-config'
    with open(tmpdir / JINA_CLOUD_CONFIG, 'w') as f:
        json.dump({'auth_token': token}, f)

    mock = mocker.Mock()
    _mock_post(mock, monkeypatch)
    _mock_get(mock, monkeypatch)

    docs = random_docs(2)
    name = random_identity()
    docs.push(name)
    DocumentArray.pull(name)

    del os.environ['JINA_HUB_ROOT']

    _get_hub_config.cache_clear()
    _get_auth_token.cache_clear()

    assert mock.call_count == 3  # 1 for push, 1 for pull, 1 for download

    _, push_kwargs = mock.call_args_list[0]
    _, pull_kwargs = mock.call_args_list[1]

    assert push_kwargs['headers'].get('Authorization') == 'token test-auth-token-env'
    assert pull_kwargs['headers'].get('Authorization') == 'token test-auth-token-env'
