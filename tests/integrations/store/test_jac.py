import multiprocessing as mp
import uuid

import hubble
import pytest

from docarray import DocumentArray
from docarray.documents import TextDoc
from docarray.store import JACDocStore
from tests.integrations.store import gen_text_docs, get_test_da, profile_memory

DA_LEN: int = 2**10
TOLERANCE_RATIO = 0.5  # Percentage of difference allowed in stream vs non-stream test
RANDOM: str = uuid.uuid4().hex[:8]


@pytest.fixture(scope='session', autouse=True)
def testing_namespace_cleanup():
    da_names = list(
        filter(
            lambda x: x.startswith('test'),
            JACDocStore.list('jac://', show_table=False),
        )
    )
    for da_name in da_names:
        JACDocStore.delete(f'jac://{da_name}')
    yield
    da_names = list(
        filter(
            lambda x: x.startswith(f'test{RANDOM}'),
            JACDocStore.list('jac://', show_table=False),
        )
    )
    for da_name in da_names:
        JACDocStore.delete(f'jac://{da_name}')


@pytest.mark.slow
@pytest.mark.internet
def test_pushpull_correct(capsys):
    DA_NAME: str = f'test{RANDOM}-pushpull-correct'
    da1 = get_test_da(DA_LEN)

    # Verbose
    da1.push(f'jac://{DA_NAME}', show_progress=True)
    da2 = DocumentArray[TextDoc].pull(f'jac://{DA_NAME}', show_progress=True)
    assert len(da1) == len(da2)
    assert all(d1.id == d2.id for d1, d2 in zip(da1, da2))
    assert all(d1.text == d2.text for d1, d2 in zip(da1, da2))

    captured = capsys.readouterr()
    assert len(captured.out) > 0
    assert len(captured.err) == 0

    # Quiet
    da2.push(f'jac://{DA_NAME}')
    da1 = DocumentArray[TextDoc].pull(f'jac://{DA_NAME}')
    assert len(da1) == len(da2)
    assert all(d1.id == d2.id for d1, d2 in zip(da1, da2))
    assert all(d1.text == d2.text for d1, d2 in zip(da1, da2))

    captured = capsys.readouterr()
    assert (
        len(captured.out) == 0
    ), 'No output should be printed when show_progress=False'
    assert len(captured.err) == 0, 'No error should be printed when show_progress=False'


@pytest.mark.slow
@pytest.mark.internet
def test_pushpull_stream_correct(capsys):
    DA_NAME_1: str = f'test{RANDOM}-pushpull-stream-correct-da1'
    DA_NAME_2: str = f'test{RANDOM}-pushpull-stream-correct-da2'

    da1 = get_test_da(DA_LEN)

    # Verbosity and correctness
    DocumentArray[TextDoc].push_stream(
        iter(da1), f'jac://{DA_NAME_1}', show_progress=True
    )
    doc_stream2 = DocumentArray[TextDoc].pull_stream(
        f'jac://{DA_NAME_1}', show_progress=True
    )

    assert all(d1.id == d2.id for d1, d2 in zip(da1, doc_stream2))
    with pytest.raises(StopIteration):
        next(doc_stream2)

    captured = capsys.readouterr()
    assert len(captured.out) > 0
    assert len(captured.err) == 0

    # Quiet and chained
    doc_stream = DocumentArray[TextDoc].pull_stream(
        f'jac://{DA_NAME_1}', show_progress=False
    )
    DocumentArray[TextDoc].push_stream(
        doc_stream, f'jac://{DA_NAME_2}', show_progress=False
    )

    captured = capsys.readouterr()
    assert (
        len(captured.out) == 0
    ), 'No output should be printed when show_progress=False'
    assert len(captured.err) == 0, 'No error should be printed when show_progress=False'


@pytest.mark.slow
@pytest.mark.internet
def test_pull_stream_vs_pull_full():
    import docarray.store.helpers

    docarray.store.helpers.CACHING_REQUEST_READER_CHUNK_SIZE = 2**10
    DA_NAME_SHORT: str = f'test{RANDOM}-pull-stream-vs-pull-full-short'
    DA_NAME_LONG: str = f'test{RANDOM}-pull-stream-vs-pull-full-long'

    DocumentArray[TextDoc].push_stream(
        gen_text_docs(DA_LEN * 1),
        f'jac://{DA_NAME_SHORT}',
        show_progress=False,
    )
    DocumentArray[TextDoc].push_stream(
        gen_text_docs(DA_LEN * 4),
        f'jac://{DA_NAME_LONG}',
        show_progress=False,
    )

    @profile_memory
    def get_total_stream(url: str):
        return sum(
            len(d.text)
            for d in DocumentArray[TextDoc].pull_stream(url, show_progress=False)
        )

    @profile_memory
    def get_total_full(url: str):
        return sum(
            len(d.text) for d in DocumentArray[TextDoc].pull(url, show_progress=False)
        )

    # A warmup is needed to get accurate memory usage comparison
    _ = get_total_stream(f'jac://{DA_NAME_SHORT}')
    short_total_stream, (_, short_stream_peak) = get_total_stream(
        f'jac://{DA_NAME_SHORT}'
    )
    long_total_stream, (_, long_stream_peak) = get_total_stream(f'jac://{DA_NAME_LONG}')

    _ = get_total_full(f'jac://{DA_NAME_SHORT}')
    short_total_full, (_, short_full_peak) = get_total_full(f'jac://{DA_NAME_SHORT}')
    long_total_full, (_, long_full_peak) = get_total_full(f'jac://{DA_NAME_LONG}')

    assert (
        short_total_stream == short_total_full
    ), 'Streamed and non-streamed pull should have similar statistics'
    assert (
        long_total_stream == long_total_full
    ), 'Streamed and non-streamed pull should have similar statistics'

    assert (
        abs(long_stream_peak - short_stream_peak) / short_stream_peak < TOLERANCE_RATIO
    ), 'Streamed memory usage should not be dependent on the size of the data'
    assert (
        abs(long_full_peak - short_full_peak) / short_full_peak > TOLERANCE_RATIO
    ), 'Full pull memory usage should be dependent on the size of the data'


@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.skip(reason='The CI account might be broken')
def test_list_and_delete():
    DA_NAME_0 = f'test{RANDOM}-list-and-delete-da0'
    DA_NAME_1 = f'test{RANDOM}-list-and-delete-da1'

    da_names = list(
        filter(
            lambda x: x.startswith(f'test{RANDOM}-list-and-delete'),
            JACDocStore.list(show_table=False),
        )
    )
    assert len(da_names) == 0

    DocumentArray[TextDoc].push(
        get_test_da(DA_LEN), f'jac://{DA_NAME_0}', show_progress=False
    )
    da_names = list(
        filter(
            lambda x: x.startswith(f'test{RANDOM}-list-and-delete'),
            JACDocStore.list(show_table=False),
        )
    )
    assert set(da_names) == {DA_NAME_0}
    DocumentArray[TextDoc].push(
        get_test_da(DA_LEN), f'jac://{DA_NAME_1}', show_progress=False
    )
    da_names = list(
        filter(
            lambda x: x.startswith(f'test{RANDOM}-list-and-delete'),
            JACDocStore.list(show_table=False),
        )
    )
    assert set(da_names) == {DA_NAME_0, DA_NAME_1}

    assert JACDocStore.delete(
        f'{DA_NAME_0}'
    ), 'Deleting an existing DA should return True'
    da_names = list(
        filter(
            lambda x: x.startswith(f'test{RANDOM}-list-and-delete'),
            JACDocStore.list(show_table=False),
        )
    )
    assert set(da_names) == {DA_NAME_1}

    with pytest.raises(
        hubble.excepts.RequestedEntityNotFoundError
    ):  # Deleting a non-existent DA without safety should raise an error
        JACDocStore.delete(f'{DA_NAME_0}', missing_ok=False)

    assert not JACDocStore.delete(
        f'{DA_NAME_0}', missing_ok=True
    ), 'Deleting a non-existent DA should return False'


@pytest.mark.slow
@pytest.mark.internet
def test_concurrent_push_pull():
    # Push to DA that is being pulled should not mess up the pull
    DA_NAME_0 = f'test{RANDOM}-concurrent-push-pull-da0'

    DocumentArray[TextDoc].push_stream(
        gen_text_docs(DA_LEN),
        f'jac://{DA_NAME_0}',
        show_progress=False,
    )

    global _task

    def _task(choice: str):
        if choice == 'push':
            DocumentArray[TextDoc].push_stream(
                gen_text_docs(DA_LEN),
                f'jac://{DA_NAME_0}',
                show_progress=False,
            )
        elif choice == 'pull':
            pull_len = sum(
                1 for _ in DocumentArray[TextDoc].pull_stream(f'jac://{DA_NAME_0}')
            )
            assert pull_len == DA_LEN
        else:
            raise ValueError(f'Unknown choice {choice}')

    with mp.get_context('fork').Pool(3) as p:
        p.map(_task, ['pull', 'push', 'pull'])
