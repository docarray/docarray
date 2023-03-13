import multiprocessing as mp
import os
import time
import uuid

import pytest

from docarray import DocumentArray
from docarray.documents import TextDoc
from tests.integrations.remotes import gen_text_docs, get_test_da, profile_memory

DA_LEN: int = 2**10
TOLERANCE_RATIO = 0.1  # Percentage of difference allowed in stream vs non-stream test
BUCKET: str = 'da-pushpull'
RANDOM: str = uuid.uuid4().hex[:8]


@pytest.fixture(scope="session")
def minio_container():
    file_dir = os.path.dirname(__file__)
    os.system(
        f"docker-compose -f {os.path.join(file_dir, 'docker-compose.yml')} up minio -d --remove-orphans"
    )
    time.sleep(1)
    yield
    os.system(
        f"docker-compose -f {os.path.join(file_dir, 'docker-compose.yml')} down --remove-orphans"
    )


@pytest.fixture(scope='session', autouse=True)
def testing_bucket(minio_container):
    import boto3
    from botocore.client import Config

    boto3.Session.resource.__defaults__ = (
        "us-east-1",
        None,
        False,
        None,
        "http://localhost:9005",
        "minioadmin",
        "minioadmin",
        None,
        Config(signature_version="s3v4"),
    )
    boto3.Session.client.__defaults__ = (
        "us-east-1",
        None,
        False,
        None,
        "http://localhost:9005",
        "minioadmin",
        "minioadmin",
        None,
        Config(signature_version="s3v4"),
    )
    # make a bucket
    s3 = boto3.resource('s3')
    s3.create_bucket(Bucket=BUCKET)

    yield
    s3.Bucket(BUCKET).objects.all().delete()
    s3.Bucket(BUCKET).delete()


def test_pushpull_correct(capsys):
    namespace_dir = f'{BUCKET}/test{RANDOM}/pushpull-correct'
    da1 = get_test_da(DA_LEN)

    # Verbose
    da1.push(f's3://{namespace_dir}/meow', show_progress=True)
    da2 = DocumentArray[TextDoc].pull(f's3://{namespace_dir}/meow', show_progress=True)
    assert len(da1) == len(da2)
    assert all(d1.id == d2.id for d1, d2 in zip(da1, da2))
    assert all(d1.text == d2.text for d1, d2 in zip(da1, da2))

    captured = capsys.readouterr()
    assert len(captured.out) > 0
    assert len(captured.err) == 0

    # Quiet
    da2.push(f's3://{namespace_dir}/meow')
    da1 = DocumentArray[TextDoc].pull(f's3://{namespace_dir}/meow')
    assert len(da1) == len(da2)
    assert all(d1.id == d2.id for d1, d2 in zip(da1, da2))
    assert all(d1.text == d2.text for d1, d2 in zip(da1, da2))

    captured = capsys.readouterr()
    assert len(captured.out) == 0
    assert len(captured.err) == 0


def test_pushpull_stream_correct(capsys):
    namespace_dir = f'{BUCKET}/test{RANDOM}/pushpull-stream-correct'
    da1 = get_test_da(DA_LEN)

    # Verbosity and correctness
    DocumentArray[TextDoc].push_stream(
        iter(da1), f's3://{namespace_dir}/meow', show_progress=True
    )
    doc_stream2 = DocumentArray[TextDoc].pull_stream(
        f's3://{namespace_dir}/meow', show_progress=True
    )

    assert all(d1.id == d2.id for d1, d2 in zip(da1, doc_stream2))
    with pytest.raises(StopIteration):
        next(doc_stream2)

    captured = capsys.readouterr()
    assert len(captured.out) > 0
    assert len(captured.err) == 0

    # Quiet and chained
    doc_stream = DocumentArray[TextDoc].pull_stream(
        f's3://{namespace_dir}/meow', show_progress=False
    )
    DocumentArray[TextDoc].push_stream(
        doc_stream, f's3://{namespace_dir}/meow2', show_progress=False
    )

    captured = capsys.readouterr()
    assert len(captured.out) == 0
    assert len(captured.err) == 0


def test_pull_stream_vs_pull_full():
    namespace_dir = f'{BUCKET}/test{RANDOM}/pull-stream-vs-pull-full'
    DocumentArray[TextDoc].push_stream(
        gen_text_docs(DA_LEN * 1),
        f's3://{namespace_dir}/meow-short',
        show_progress=False,
    )
    DocumentArray[TextDoc].push_stream(
        gen_text_docs(DA_LEN * 4),
        f's3://{namespace_dir}/meow-long',
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
    _ = get_total_stream(f's3://{namespace_dir}/meow-short')
    short_total_stream, (_, short_stream_peak) = get_total_stream(
        f's3://{namespace_dir}/meow-short'
    )
    long_total_stream, (_, long_stream_peak) = get_total_stream(
        f's3://{namespace_dir}/meow-long'
    )

    _ = get_total_full(f's3://{namespace_dir}/meow-short')
    short_total_full, (_, short_full_peak) = get_total_full(
        f's3://{namespace_dir}/meow-short'
    )
    long_total_full, (_, long_full_peak) = get_total_full(
        f's3://{namespace_dir}/meow-long'
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


def test_list_and_delete():
    namespace_dir = f'{BUCKET}/test{RANDOM}/list-and-delete'

    da_names = DocumentArray.list(f's3://{namespace_dir}', show_table=False)
    assert len(da_names) == 0

    DocumentArray[TextDoc].push_stream(
        gen_text_docs(DA_LEN), f's3://{namespace_dir}/meow', show_progress=False
    )
    da_names = DocumentArray.list(f's3://{namespace_dir}', show_table=False)
    assert set(da_names) == {'meow'}
    DocumentArray[TextDoc].push_stream(
        gen_text_docs(DA_LEN), f's3://{namespace_dir}/woof', show_progress=False
    )
    da_names = DocumentArray.list(f's3://{namespace_dir}', show_table=False)
    assert set(da_names) == {'meow', 'woof'}

    assert DocumentArray.delete(
        f's3://{namespace_dir}/meow'
    ), 'Deleting an existing DA should return True'
    da_names = DocumentArray.list(f's3://{namespace_dir}', show_table=False)
    assert set(da_names) == {'woof'}

    with pytest.raises(
        ValueError
    ):  # Deleting a non-existent DA without safety should raise an error
        DocumentArray.delete(f's3://{namespace_dir}/meow')
    assert not DocumentArray.delete(
        f's3://{namespace_dir}/meow', missing_ok=True
    ), 'Deleting a non-existent DA should return False'


def test_concurrent_push_pull():
    # Push to DA that is being pulled should not mess up the pull
    namespace_dir = f'{BUCKET}/test{RANDOM}/concurrent-push-pull'

    DocumentArray[TextDoc].push_stream(
        gen_text_docs(DA_LEN),
        f's3://{namespace_dir}/da0',
        show_progress=False,
    )

    global _task

    def _task(choice: str):
        if choice == 'push':
            DocumentArray[TextDoc].push_stream(
                gen_text_docs(DA_LEN),
                f's3://{namespace_dir}/da0',
                show_progress=False,
            )
        elif choice == 'pull':
            pull_len = sum(
                1
                for _ in DocumentArray[TextDoc].pull_stream(f's3://{namespace_dir}/da0')
            )
            assert pull_len == DA_LEN
        else:
            raise ValueError(f'Unknown choice {choice}')

    with mp.get_context('fork').Pool(3) as p:
        p.map(_task, ['pull', 'push', 'pull'])


@pytest.mark.skip(reason='Not Applicable')
def test_concurrent_push():
    """
    Amazon S3 does not support object locking for concurrent writers.
    If two PUT requests are simultaneously made to the same key, the request with the latest timestamp wins.
    However, there is no way for the processes to know if they are the latest or not.

    https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html#ConsistencyModel
    """
    pass
