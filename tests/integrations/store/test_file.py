import multiprocessing as mp
from pathlib import Path

import pytest

from docarray import DocList
from docarray.documents import TextDoc
from docarray.store.file import ConcurrentPushException, FileDocStore
from docarray.utils._internal.cache import _get_cache_path
from docarray.utils._internal.pydantic import is_pydantic_v2
from tests.integrations.store import gen_text_docs, get_test_da, profile_memory

DA_LEN: int = 2**10
TOLERANCE_RATIO = 0.1  # Percentage of difference allowed in stream vs non-stream test


def test_path_resolution():
    assert FileDocStore._abs_filepath('meow') == _get_cache_path() / 'meow'
    assert FileDocStore._abs_filepath('/meow') == Path('/meow')
    assert FileDocStore._abs_filepath('~/meow') == Path.home() / 'meow'
    assert FileDocStore._abs_filepath('./meow') == Path.cwd() / 'meow'
    assert FileDocStore._abs_filepath('../meow') == Path.cwd().parent / 'meow'


def test_pushpull_correct(capsys, tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    namespace_dir = tmp_path
    da1 = get_test_da(DA_LEN)

    # Verbose
    da1.push(f'file://{namespace_dir}/meow', show_progress=True)
    da2 = DocList[TextDoc].pull(f'file://{namespace_dir}/meow', show_progress=True)
    assert len(da1) == len(da2)
    assert all(d1.id == d2.id for d1, d2 in zip(da1, da2))
    assert all(d1.text == d2.text for d1, d2 in zip(da1, da2))

    captured = capsys.readouterr()
    assert len(captured.out) > 0
    assert len(captured.err) == 0

    # Quiet
    da2.push(f'file://{namespace_dir}/meow')
    da1 = DocList[TextDoc].pull(f'file://{namespace_dir}/meow')
    assert len(da1) == len(da2)
    assert all(d1.id == d2.id for d1, d2 in zip(da1, da2))
    assert all(d1.text == d2.text for d1, d2 in zip(da1, da2))

    captured = capsys.readouterr()
    assert len(captured.out) == 0
    assert len(captured.err) == 0


def test_pushpull_stream_correct(capsys, tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    namespace_dir = tmp_path
    da1 = get_test_da(DA_LEN)

    # Verbosity and correctness
    DocList[TextDoc].push_stream(
        iter(da1), f'file://{namespace_dir}/meow', show_progress=True
    )
    doc_stream2 = DocList[TextDoc].pull_stream(
        f'file://{namespace_dir}/meow', show_progress=True
    )

    assert all(d1.id == d2.id for d1, d2 in zip(da1, doc_stream2))
    with pytest.raises(StopIteration):
        next(doc_stream2)

    captured = capsys.readouterr()
    assert len(captured.out) > 0
    assert len(captured.err) == 0

    # Quiet and chained
    doc_stream = DocList[TextDoc].pull_stream(
        f'file://{namespace_dir}/meow', show_progress=False
    )
    DocList[TextDoc].push_stream(
        doc_stream, f'file://{namespace_dir}/meow2', show_progress=False
    )

    captured = capsys.readouterr()
    assert len(captured.out) == 0
    assert len(captured.err) == 0


# for some reason this test is failing with pydantic v2
@pytest.mark.skipif(is_pydantic_v2, reason="Not working with pydantic v2 for now")
@pytest.mark.slow
def test_pull_stream_vs_pull_full(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    namespace_dir = tmp_path
    DocList[TextDoc].push_stream(
        gen_text_docs(DA_LEN * 1),
        f'file://{namespace_dir}/meow-short',
        show_progress=False,
    )
    DocList[TextDoc].push_stream(
        gen_text_docs(DA_LEN * 4),
        f'file://{namespace_dir}/meow-long',
        show_progress=False,
    )

    @profile_memory
    def get_total_stream(url: str):
        return sum(
            len(d.text) for d in DocList[TextDoc].pull_stream(url, show_progress=False)
        )

    @profile_memory
    def get_total_full(url: str):
        return sum(len(d.text) for d in DocList[TextDoc].pull(url, show_progress=False))

    # A warmup is needed to get accurate memory usage comparison
    _ = get_total_stream(f'file://{namespace_dir}/meow-short')
    short_total_stream, (_, short_stream_peak) = get_total_stream(
        f'file://{namespace_dir}/meow-short'
    )
    long_total_stream, (_, long_stream_peak) = get_total_stream(
        f'file://{namespace_dir}/meow-long'
    )

    _ = get_total_full(f'file://{namespace_dir}/meow-short')
    short_total_full, (_, short_full_peak) = get_total_full(
        f'file://{namespace_dir}/meow-short'
    )
    long_total_full, (_, long_full_peak) = get_total_full(
        f'file://{namespace_dir}/meow-long'
    )

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


def test_list_and_delete(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    namespace_dir = str(tmp_path)

    da_names = FileDocStore.list(namespace_dir, show_table=False)
    assert len(da_names) == 0

    DocList[TextDoc].push_stream(
        gen_text_docs(DA_LEN), f'file://{namespace_dir}/meow', show_progress=False
    )
    da_names = FileDocStore.list(namespace_dir, show_table=False)
    assert set(da_names) == {'meow'}
    DocList[TextDoc].push_stream(
        gen_text_docs(DA_LEN), f'file://{namespace_dir}/woof', show_progress=False
    )
    da_names = FileDocStore.list(namespace_dir, show_table=False)
    assert set(da_names) == {'meow', 'woof'}

    assert FileDocStore.delete(
        f'{namespace_dir}/meow'
    ), 'Deleting an existing DA should return True'
    da_names = FileDocStore.list(namespace_dir, show_table=False)
    assert set(da_names) == {'woof'}

    with pytest.raises(
        FileNotFoundError
    ):  # Deleting a non-existent DA without safety should raise an error
        FileDocStore.delete(f'{namespace_dir}/meow', missing_ok=False)

    assert not FileDocStore.delete(
        f'{namespace_dir}/meow', missing_ok=True
    ), 'Deleting a non-existent DA should return False'


def test_concurrent_push_pull(tmp_path: Path):
    # Push to DA that is being pulled should not mess up the pull
    tmp_path.mkdir(parents=True, exist_ok=True)
    namespace_dir = tmp_path

    DocList[TextDoc].push_stream(
        gen_text_docs(DA_LEN),
        f'file://{namespace_dir}/da0',
        show_progress=False,
    )

    global _task

    def _task(choice: str):
        if choice == 'push':
            DocList[TextDoc].push_stream(
                gen_text_docs(DA_LEN),
                f'file://{namespace_dir}/da0',
                show_progress=False,
            )
        elif choice == 'pull':
            pull_len = sum(
                1 for _ in DocList[TextDoc].pull_stream(f'file://{namespace_dir}/da0')
            )
            assert pull_len == DA_LEN
        else:
            raise ValueError(f'Unknown choice {choice}')

    with mp.get_context('fork').Pool(3) as p:
        p.map(_task, ['pull', 'push', 'pull'])


@pytest.mark.slow
def test_concurrent_push(tmp_path: Path):
    # Double push should fail the second push
    import time

    tmp_path.mkdir(parents=True, exist_ok=True)
    namespace_dir = tmp_path

    DocList[TextDoc].push_stream(
        gen_text_docs(DA_LEN),
        f'file://{namespace_dir}/da0',
        show_progress=False,
    )

    def _slowdown_iterator(iterator):
        for i, e in enumerate(iterator):
            yield e
            if i % (DA_LEN // 100) == 0:
                time.sleep(0.01)

    global _push

    def _push(choice: str):
        if choice == 'slow':
            DocList[TextDoc].push_stream(
                _slowdown_iterator(gen_text_docs(DA_LEN)),
                f'file://{namespace_dir}/da0',
                show_progress=False,
            )
            return True
        elif choice == 'cold_start':
            try:
                time.sleep(0.1)
                DocList[TextDoc].push_stream(
                    gen_text_docs(DA_LEN),
                    f'file://{namespace_dir}/da0',
                    show_progress=False,
                )
                return True
            except ConcurrentPushException:
                return False
        else:
            raise ValueError(f'Unknown choice {choice}')

    with mp.get_context('fork').Pool(3) as p:
        results = p.map(_push, ['cold_start', 'slow', 'cold_start'])
    assert results == [False, True, False]
