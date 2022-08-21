import multiprocessing
import time

import pytest

from docarray import Document, DocumentArray
from docarray.array.mixins.post import _parse_host
from docarray.helper import random_port


@pytest.mark.parametrize(
    'host, expected_on, expected_host, expected_port, expected_version, expected_scheme',
    [
        (
            'grpc://192.168.0.123:8080/index',
            '/index',
            'grpc://192.168.0.123',
            8080,
            None,
            'grpc',
        ),
        (
            'ws://192.168.0.123:80/encode',
            '/encode',
            'ws://192.168.0.123',
            80,
            None,
            'ws',
        ),
        (
            'http://192.168.192.123:8080/index',
            '/index',
            'http://192.168.192.123',
            8080,
            None,
            'http',
        ),
        (
            'jinahub://Hello/endpoint',
            '/endpoint',
            'jinahub://Hello',
            None,
            None,
            'jinahub',
        ),
        (
            'jinahub+docker://Hello/index',
            '/index',
            'jinahub+docker://Hello',
            None,
            None,
            'jinahub+docker',
        ),
        (
            'jinahub+docker://Hello/v0.0.1/search',
            '/search',
            'jinahub+docker://Hello/v0.0.1',
            None,
            'v0.0.1',
            'jinahub+docker',
        ),
        (
            'jinahub+docker://Hello/latest/index',
            '/index',
            'jinahub+docker://Hello/latest',
            None,
            'latest',
            'jinahub+docker',
        ),
        (
            'jinahub+docker://Hello/v0.0.1-cpu/index',
            '/index',
            'jinahub+docker://Hello/v0.0.1-cpu',
            None,
            'v0.0.1-cpu',
            'jinahub+docker',
        ),
        (
            'jinahub+docker://Hello/v0.5-gpu/index',
            '/index',
            'jinahub+docker://Hello/v0.5-gpu',
            None,
            'v0.5-gpu',
            'jinahub+docker',
        ),
        (
            'jinahub+sandbox://Hello/index',
            '/index',
            'jinahub+sandbox://Hello',
            None,
            None,
            'jinahub+sandbox',
        ),
    ],
)
def test_parse_host(
    host, expected_on, expected_host, expected_port, expected_version, expected_scheme
):
    parsed_host = _parse_host(host)
    assert parsed_host.on == expected_on
    assert parsed_host.host == expected_host
    assert parsed_host.port == expected_port
    assert parsed_host.version == expected_version
    assert parsed_host.scheme == expected_scheme


@pytest.mark.parametrize(
    'conn_config',
    [
        (dict(protocol='grpc'), 'grpc://127.0.0.1:$port/'),
        (dict(protocol='grpc'), 'grpc://127.0.0.1:$port'),
        (dict(protocol='websocket'), 'ws://127.0.0.1:$port'),
        # (dict(protocol='http'), 'http://127.0.0.1:$port'),  this somehow does not work on GH workflow
    ],
)
@pytest.mark.parametrize('show_pbar', [True, False])
@pytest.mark.parametrize('batch_size', [None, 1, 10])
def test_post_to_a_flow(show_pbar, conn_config, batch_size):
    from jina import Flow

    p = random_port()
    da = DocumentArray.empty(100)
    with Flow(**{**conn_config[0], 'port': p}):
        da.post(conn_config[1].replace('$port', str(p)), batch_size=batch_size)


@pytest.mark.parametrize(
    'hub_uri',
    [
        'jinahub://Hello',
        'jinahub+sandbox://Hello',
        # 'jinahub+docker://Hello',  this somehow does not work on GH workflow
    ],
)
def test_post_with_jinahub(hub_uri):
    da = DocumentArray.empty(100)
    da.post(hub_uri)

    assert isinstance(Document().post(hub_uri), Document)


def test_post_bad_scheme():
    da = DocumentArray.empty(100)
    with pytest.raises(ValueError):
        da.post('haha')


def test_endpoint():
    from jina import Executor, Flow, requests

    class MyExec(Executor):
        @requests(on='/foo')
        def foo(self, docs: DocumentArray, **kwargs):
            docs.texts = ['foo'] * len(docs)

        @requests(on='/bar')
        def bar(self, docs: DocumentArray, **kwargs):
            docs.texts = ['bar'] * len(docs)

    def start_flow(stop_event, **kwargs):
        """start a blocking Flow."""
        with Flow(**kwargs).add(uses=MyExec) as f:
            f.block(stop_event=stop_event)

    e = multiprocessing.Event()  # create new Event

    p = random_port()
    t = multiprocessing.Process(
        name='Blocked-Flow', target=start_flow, args=(e,), kwargs={'port': p}
    )
    t.start()

    time.sleep(5)
    N = 100
    da = DocumentArray.empty(N)
    try:
        assert da.post(f'grpc://127.0.0.1:{p}/')[:, 'text'] == [''] * N
        assert da.post(f'grpc://127.0.0.1:{p}/foo').texts == ['foo'] * N
        assert da.post(f'grpc://127.0.0.1:{p}/bar').texts == ['bar'] * N
    except:
        raise
    finally:
        e.set()
        t.join()
