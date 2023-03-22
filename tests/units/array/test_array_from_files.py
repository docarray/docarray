import pytest

from docarray import BaseDocument, DocumentArray
from docarray.array.array.io import from_files
from docarray.documents import ImageDoc
from docarray.typing import TextUrl
from tests import TOYDATA_DIR
from tests.units.typing.url.test_image_url import PATH_TO_IMAGE_DATA


@pytest.mark.parametrize(
    'patterns, recursive, size, sampling_rate',
    [
        (f'{PATH_TO_IMAGE_DATA}/*.*', True, None, None),
        (f'{PATH_TO_IMAGE_DATA}/*.*', False, None, None),
        (f'{PATH_TO_IMAGE_DATA}/*.*', True, 2, None),
        (f'{PATH_TO_IMAGE_DATA}/*.*', True, None, 0.5),
    ],
)
def test_from_files(patterns, recursive, size, sampling_rate):
    da = DocumentArray[ImageDoc](
        list(
            from_files(
                url_field='url',
                doc_type=ImageDoc,
                patterns=patterns,
                recursive=recursive,
                size=size,
                sampling_rate=sampling_rate,
            )
        )
    )
    if size:
        assert len(da) <= size
    for doc in da:
        doc.summary()
        assert doc.url is not None


@pytest.mark.parametrize(
    'patterns',
    [
        (f'{TOYDATA_DIR}/*.txt'),
    ],
)
def test_from_files_with_storing_file_content(patterns):
    class MyDoc(BaseDocument):
        url: TextUrl
        some_text: str

    da = DocumentArray[MyDoc](
        list(
            from_files(
                patterns=patterns,
                doc_type=MyDoc,
                url_field='url',
                content_field='some_text',
            )
        )
    )
    assert len(da) == 1
    for doc in da:
        assert isinstance(doc, MyDoc)
        assert doc.url is not None
        assert doc.some_text is not None


@pytest.mark.parametrize(
    'patterns',
    [
        (f'{TOYDATA_DIR}/*.txt'),
    ],
)
def test_document_array_from_files(patterns):
    class MyDoc(BaseDocument):
        url: TextUrl
        some_text: str

    da = DocumentArray[MyDoc].from_files(
        patterns=patterns,
        url_field='url',
        content_field='some_text',
    )

    assert len(da) == 1
    for doc in da:
        assert isinstance(doc, MyDoc)
        assert doc.url is not None
        assert doc.some_text is not None


def test_from_files_with_nested_fields():
    class MyDoc(BaseDocument):
        url: TextUrl
        some_text: str

    class NestedDoc(BaseDocument):
        my_doc: MyDoc

    da = DocumentArray[NestedDoc].from_files(
        patterns=f'{TOYDATA_DIR}/*.txt',
        url_field='my_doc__url',
        content_field='my_doc__some_text',
    )

    assert len(da) == 1
    for doc in da:
        assert isinstance(doc, NestedDoc)
        assert doc.my_doc.url is not None
        assert doc.my_doc.some_text is not None
