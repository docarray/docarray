import os
import sys

import numpy as np
import pytest

from docarray import Document
from docarray.document.generators import from_files

__windows__ = sys.platform == 'win32'

cur_dir = os.path.dirname(os.path.abspath(__file__))


def test_video_convert_pipe(pytestconfig, tmpdir):
    num_d = 0
    fname = str(tmpdir / f'tmp{num_d}.mp4')
    d = Document(uri=os.path.join(cur_dir, 'toydata/mov_bbb.mp4'))
    d.load_uri_to_video_tensor()
    d.save_video_tensor_to_file(fname)
    assert os.path.exists(fname)


def test_audio_convert_pipe(pytestconfig, tmpdir):
    num_d = 0
    for d in from_files(f'{cur_dir}/toydata/*.wav'):
        fname = str(tmpdir / f'tmp{num_d}.wav')
        d.load_uri_to_audio_tensor()
        d.tensor = d.tensor[::-1]
        d.save_audio_tensor_to_file(fname)
        assert os.path.exists(fname)
        num_d += 1
    assert num_d


def test_image_convert_pipe(pytestconfig):
    for d in from_files(f'{pytestconfig.rootdir}/.github/**/*.png'):
        (
            d.load_uri_to_image_tensor()
            .convert_uri_to_datauri()
            .set_image_tensor_shape((64, 64))
            .set_image_tensor_normalization()
            .set_image_tensor_channel_axis(-1, 0)
        )
        assert d.tensor.shape == (3, 64, 64)
        assert d.uri


def test_uri_to_tensor():
    doc = Document(uri=os.path.join(cur_dir, 'toydata/test.png'))
    doc.load_uri_to_image_tensor()
    assert isinstance(doc.tensor, np.ndarray)
    assert doc.tensor.shape == (85, 152, 3)  # h,w,c
    assert doc.mime_type == 'image/png'


def test_datauri_to_tensor():
    doc = Document(uri=os.path.join(cur_dir, 'toydata/test.png'))
    doc.convert_uri_to_datauri()
    assert not doc.tensor
    assert doc.mime_type == 'image/png'


def test_blob_to_tensor():
    doc = Document(uri=os.path.join(cur_dir, 'toydata/test.png'))
    doc.load_uri_to_blob()
    doc.convert_blob_to_image_tensor()
    assert isinstance(doc.tensor, np.ndarray)
    assert doc.mime_type == 'image/png'
    assert doc.tensor.shape == (85, 152, 3)  # h,w,c


def test_convert_blob_to_tensor():
    rand_state = np.random.RandomState(0)
    array = rand_state.random([10, 10])
    doc = Document(content=array.tobytes())
    assert doc.content_type == 'blob'
    intialiazed_blob = doc.blob

    doc.convert_blob_to_tensor()
    assert doc.content_type == 'tensor'
    converted_blob_in_one_of = doc.blob
    assert intialiazed_blob != converted_blob_in_one_of
    np.testing.assert_almost_equal(doc.content.reshape([10, 10]), array)


@pytest.mark.parametrize('shape, channel_axis', [((3, 32, 32), 0), ((32, 32, 3), -1)])
def test_image_normalize(shape, channel_axis):
    doc = Document(content=np.random.randint(0, 255, shape, dtype=np.uint8))
    doc.set_image_tensor_normalization(channel_axis=channel_axis)
    assert doc.tensor.ndim == 3
    assert doc.tensor.shape == shape
    assert doc.tensor.dtype == np.float32


@pytest.mark.parametrize(
    'arr_size, channel_axis, height, width',
    [
        ([32, 28, 3], -1, 32, 28),  # h, w, c (rgb)
        ([3, 32, 28], 0, 32, 28),  # c, h, w  (rgb)
        ([1, 32, 28], 0, 32, 28),  # c, h, w, (greyscale)
        ([32, 28, 1], -1, 32, 28),  # h, w, c, (greyscale)
    ],
)
@pytest.mark.parametrize('format', ['png', 'jpeg'])
def test_convert_image_tensor_to_uri(arr_size, channel_axis, width, height, format):
    doc = Document(content=np.random.randint(0, 255, arr_size))
    assert doc.tensor.any()
    assert not doc.uri
    doc.set_image_tensor_shape(channel_axis=channel_axis, shape=(width, height))
    org_size = doc.tensor.shape
    doc.set_image_tensor_resample(0.5, channel_axis=channel_axis)
    for n, o in zip(doc.tensor.shape, org_size):
        if o not in (1, 3):
            assert n == 0.5 * o
    doc.set_image_tensor_resample(2, channel_axis=channel_axis)
    for n, o in zip(doc.tensor.shape, org_size):
        if o not in (1, 3):
            assert n == o

    doc.convert_image_tensor_to_uri(channel_axis=channel_axis, image_format=format)
    assert doc.uri.startswith(f'data:image/{format};base64,')
    assert doc.mime_type == f'image/{format}'
    assert doc.tensor.any()  # assure after conversion tensor still exist.


def test_convert_pillow_image_to_uri():
    doc = Document(content=np.random.randint(0, 255, [32, 32, 3]))
    assert doc.tensor.any()
    assert not doc.uri
    import PIL.Image

    p = PIL.Image.fromarray(doc.tensor, mode='RGB')
    doc.load_pil_image_to_datauri(p)
    assert doc.uri.startswith(f'data:image/png;base64,')
    assert doc.mime_type == 'image/png'
    assert doc.tensor.any()  # assure after conversion tensor still exist.


@pytest.mark.xfail(
    condition=__windows__, reason='x-python is not detected on windows CI'
)
@pytest.mark.parametrize(
    'uri, mimetype',
    [
        (__file__, 'text/x-python'),
        ('http://google.com/index.html', 'text/html'),
        ('https://google.com/index.html', 'text/html'),
    ],
)
def test_convert_uri_to_blob(uri, mimetype):
    d = Document(uri=uri)
    assert not d.blob
    d.load_uri_to_blob()
    assert d.blob
    assert d.mime_type == mimetype


@pytest.mark.parametrize(
    'converter', ['convert_blob_to_datauri', 'convert_content_to_datauri']
)
def test_convert_blob_to_uri(converter):
    d = Document(content=open(__file__).read().encode(), mime_type='text/x-python')
    assert d.blob
    getattr(d, converter)()
    assert d.uri.startswith('data:text/x-python;')


@pytest.mark.parametrize(
    'converter', ['convert_text_to_datauri', 'convert_content_to_datauri']
)
def test_convert_text_to_uri(converter):
    d = Document(content=open(__file__).read(), mime_type='text/plain')
    assert d.text
    getattr(d, converter)()
    assert d.uri.startswith('data:text/plain;')


@pytest.mark.xfail(
    condition=__windows__, reason='x-python is not detected on windows CI'
)
@pytest.mark.parametrize(
    'uri, mimetype',
    [
        pytest.param(
            __file__,
            'text/x-python',
            marks=pytest.mark.xfail(
                condition=__windows__, reason='x-python is not detected on windows CI'
            ),
        ),
        ('http://google.com/index.html', 'text/html'),
        ('https://google.com/index.html', 'text/html'),
    ],
)
def test_convert_uri_to_text(uri, mimetype):
    doc = Document(uri=uri, mime_type=mimetype)
    doc.load_uri_to_text()
    if mimetype == 'text/html':
        assert '<!doctype html>' in doc.text
    elif mimetype == 'text/x-python':
        text_from_file = open(__file__).read()
        assert doc.text == text_from_file


def test_convert_text_to_uri_and_back():
    text_from_file = open(__file__).read()
    doc = Document(content=text_from_file)
    assert doc.text
    assert not doc.mime_type
    doc.convert_text_to_datauri()
    doc.load_uri_to_text()
    assert doc.mime_type == 'text/plain'
    assert doc.text == text_from_file


def test_convert_text_diff_encoding(tmpfile):
    otext = 'testä'
    text = otext.encode('iso8859')
    with open(tmpfile, 'wb') as fp:
        fp.write(text)
    with pytest.raises(UnicodeDecodeError):
        d = Document(uri=str(tmpfile)).load_uri_to_text()

    d = Document(uri=str(tmpfile)).load_uri_to_text(charset='iso8859')
    assert d.text == otext

    with open(tmpfile, 'w', encoding='iso8859') as fp:
        fp.write(otext)
    with pytest.raises(UnicodeDecodeError):
        d = Document(uri=str(tmpfile)).load_uri_to_text()

    d = Document(uri=str(tmpfile)).load_uri_to_text(charset='iso8859')
    assert d.text == otext


def test_convert_content_to_uri():
    d = Document(content=np.random.random([10, 10]))
    with pytest.raises(NotImplementedError):
        d.convert_content_to_datauri()


@pytest.mark.parametrize(
    'uri, mimetype',
    [
        (__file__, 'text/x-python'),
        ('http://google.com/index.html', 'text/html'),
        ('https://google.com/index.html', 'text/html'),
    ],
)
def test_convert_uri_to_data_uri(uri, mimetype):
    doc = Document(uri=uri, mime_type=mimetype)
    doc.convert_uri_to_datauri()
    assert doc.uri.startswith(f'data:{mimetype}')
    assert doc.mime_type == mimetype


@pytest.mark.parametrize(
    'uri, chunk_num',
    [
        (os.path.join(cur_dir, 'toydata/test.glb'), 1),
        (
            'https://github.com/jina-ai/docarray/raw/main/tests/unit/document/toydata/test.glb',
            1,
        ),
    ],
)
def test_glb_converters(uri, chunk_num):
    doc = Document(uri=uri)
    doc.load_uri_to_point_cloud_tensor(2000)
    assert doc.tensor.shape == (2000, 3)
    assert isinstance(doc.tensor, np.ndarray)

    doc.load_uri_to_point_cloud_tensor(2000, as_chunks=True)
    assert len(doc.chunks) == chunk_num
    assert doc.chunks[0].tensor.shape == (2000, 3)


@pytest.mark.parametrize(
    'uri',
    [
        os.path.join(cur_dir, 'toydata/test.glb'),
        'https://github.com/jina-ai/docarray/raw/main/tests/unit/document/toydata/test.glb',
    ],
)
def test_glb_not_removing_docs_text_attribute(uri):
    doc = Document(uri=uri, text='hello')
    doc.load_uri_to_point_cloud_tensor(2000)

    assert doc.text == 'hello'
