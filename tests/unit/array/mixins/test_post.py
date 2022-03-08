import multiprocessing
import time

import pytest

from docarray import DocumentArray
from docarray.helper import random_port


@pytest.mark.parametrize(
    'conn_config',
    [
        (dict(protocol='grpc'), 'grpc://0.0.0.0:$port/'),
        (dict(protocol='grpc'), 'grpc://0.0.0.0:$port'),
        (dict(protocol='websocket'), 'websocket://0.0.0.0:$port'),
        (dict(protocol='http'), 'http://0.0.0.0:$port'),
    ],
)
@pytest.mark.parametrize('show_pbar', [True, False])
def test_post_to_a_flow(show_pbar, conn_config):
    from jina import Flow

    def start_flow(stop_event, **kwargs):
        """start a blocking Flow."""
        with Flow(**kwargs) as f:
            f.block(stop_event=stop_event)
            print('bye')

    e = multiprocessing.Event()  # create new Event

    p = random_port()
    t = multiprocessing.Process(
        name='Blocked-Flow',
        target=start_flow,
        args=(e,),
        kwargs={**conn_config[0], 'port': p},
    )
    t.start()

    time.sleep(1)

    da = DocumentArray.empty(100)
    try:
        da.post(conn_config[1].replace('$port', str(p)))
    except:
        raise
    finally:
        e.set()
        t.join()
        time.sleep(1)


@pytest.mark.parametrize(
    'hub_uri', ['jinahub://Hello', 'jinahub+docker://Hello', 'jinahub+sandbox://Hello']
)
def test_post_with_jinahub(hub_uri):
    da = DocumentArray.empty(100)
    da.post(hub_uri)


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

    time.sleep(1)
    N = 100
    da = DocumentArray.empty(N)
    try:
        assert da.post(f'grpc://0.0.0.0:{p}/')[:, 'text'] == [''] * N
        assert da.post(f'grpc://0.0.0.0:{p}/foo').texts == ['foo'] * N
        assert da.post(f'grpc://0.0.0.0:{p}/bar').texts == ['bar'] * N
    except:
        raise
    finally:
        e.set()
        t.join()
