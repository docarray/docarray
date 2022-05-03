import multiprocessing
import time

import pytest

from docarray import DocumentArray, Document
from docarray.helper import random_port


'''@pytest.mark.parametrize(
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
    from jina import Executor, requests, Flow

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
'''
