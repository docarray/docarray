import base64
import os
import pickle
from typing import List, TypeVar

import numpy as np
import pytest

from docarray import Document, DocumentArray
from docarray.array.chunk import ChunkArray
from docarray.dataclasses import dataclass, field
from docarray.typing import Image, Text, Audio, Video, Mesh, Tabular, Blob, JSON
from docarray.dataclasses.getter import image_getter
from docarray.dataclasses.enums import DocumentMetadata, AttributeType

cur_dir = os.path.dirname(os.path.abspath(__file__))

AUDIO_URI = os.path.join(cur_dir, 'toydata/hello.wav')
IMAGE_URI = os.path.join(cur_dir, 'toydata/test.png')
VIDEO_URI = os.path.join(cur_dir, 'toydata/mov_bbb.mp4')
MESH_URI = os.path.join(cur_dir, 'toydata/test.glb')
TABULAR_URI = os.path.join(cur_dir, 'toydata/docs.csv')


def _assert_doc_schema(doc, schema):
    for field, attr_type, _type, position in schema:
        assert (
            doc._metadata[DocumentMetadata.MULTI_MODAL_SCHEMA][field]['attribute_type']
            == attr_type
        )
        assert (
            doc._metadata[DocumentMetadata.MULTI_MODAL_SCHEMA][field]['type'] == _type
        )
        if position is not None:
            assert (
                doc._metadata[DocumentMetadata.MULTI_MODAL_SCHEMA][field]['position']
                == position
            )
        else:
            assert (
                'position'
                not in doc._metadata[DocumentMetadata.MULTI_MODAL_SCHEMA][field]
            )


def test_type_annotation():
    @dataclass
    class MMDoc:
        f1: Video
        f2: Mesh
        f3: Blob
        f4: Tabular = None

    m1 = MMDoc(f1=VIDEO_URI, f2=MESH_URI, f3=b'hello', f4=TABULAR_URI)

    m_r = MMDoc(Document(m1))

    assert m_r == m1

    # test direct tensor assignment
    m2 = MMDoc(
        f1=np.random.random([10, 10]),
        f2=np.random.random([10, 10]),
        f3=MESH_URI,  # intentional, to test file path as binary
    )

    m_r = MMDoc(Document(m2))

    assert m_r == m2


def test_simple():
    @dataclass
    class MMDocument:
        title: Text
        image: Image
        version: int

    obj = MMDocument(title='hello world', image=np.random.rand(10, 10, 3), version=20)
    doc = Document(obj)
    assert doc.chunks[0].text == 'hello world'
    assert doc.chunks[1].tensor.shape == (10, 10, 3)
    assert doc.tags['version'] == 20

    assert doc.is_multimodal

    expected_schema = [
        ('title', AttributeType.DOCUMENT, 'Text', 0),
        ('image', AttributeType.DOCUMENT, 'Image', 1),
        ('version', AttributeType.PRIMITIVE, 'int', None),
    ]
    _assert_doc_schema(doc, expected_schema)

    translated_obj = MMDocument(doc)
    assert translated_obj == obj


def test_simple_default():
    @dataclass
    class MMDocument:
        title: Text = 'hello world'
        image: Image = IMAGE_URI
        version: int = 1

    obj = MMDocument()
    assert obj.title == 'hello world'
    assert obj.image == IMAGE_URI
    assert obj.version == 1
    doc = Document(obj)
    assert doc.chunks[0].text == 'hello world'
    assert doc.chunks[1].tensor.shape == (85, 152, 3)
    assert doc.tags['version'] == 1

    obj = MMDocument(title='custom text')
    assert obj.title == 'custom text'
    assert obj.image == IMAGE_URI
    assert obj.version == 1


def test_nested():
    @dataclass
    class SubDocument:
        image: Image
        date: str
        version: float

    @dataclass
    class MMDocument:
        sub_doc: SubDocument
        value: str
        image: Image

    obj = MMDocument(
        sub_doc=SubDocument(image=IMAGE_URI, date='10.03.2022', version=1.5),
        image=np.random.rand(10, 10, 3),
        value='abc',
    )

    doc = Document(obj)
    assert doc.tags['value'] == 'abc'

    assert doc.chunks[0].tags['date'] == '10.03.2022'
    assert doc.chunks[0].tags['version'] == 1.5
    assert doc.chunks[0].chunks[0].tensor.shape == (85, 152, 3)

    assert doc.chunks[1].tensor.shape == (10, 10, 3)

    assert doc.is_multimodal

    expected_schema = [
        ('sub_doc', AttributeType.NESTED, 'SubDocument', 0),
        ('image', AttributeType.DOCUMENT, 'Image', 1),
        ('value', AttributeType.PRIMITIVE, 'str', None),
    ]
    _assert_doc_schema(doc, expected_schema)

    translated_obj = MMDocument(doc)
    assert translated_obj == obj


def test_with_tags():
    @dataclass
    class MMDocument:
        attr1: str
        attr2: int
        attr3: float
        attr4: bool
        attr5: bytes

    obj = MMDocument(attr1='123', attr2=10, attr3=1.1, attr4=True, attr5=b'ab1234')

    doc = Document(obj)
    assert doc.tags['attr1'] == '123'
    assert doc.tags['attr2'] == 10
    assert doc.tags['attr3'] == 1.1
    assert doc.tags['attr4'] == True
    assert doc.tags['attr5'] == base64.b64encode(b'ab1234').decode()

    assert doc.is_multimodal

    expected_schema = [
        ('attr1', AttributeType.PRIMITIVE, 'str', None),
        ('attr2', AttributeType.PRIMITIVE, 'int', None),
        ('attr3', AttributeType.PRIMITIVE, 'float', None),
        ('attr4', AttributeType.PRIMITIVE, 'bool', None),
        ('attr5', AttributeType.PRIMITIVE, 'bytes', None),
    ]
    _assert_doc_schema(doc, expected_schema)

    translated_obj = MMDocument(doc)
    assert translated_obj == obj


def test_iterable_doc():
    @dataclass
    class SocialPost:
        comments: List[str]
        ratings: List[int]
        images: List[Image]

    obj = SocialPost(
        comments=['hello world', 'goodbye world'],
        ratings=[1, 5, 4, 2],
        images=[np.random.rand(10, 10, 3) for _ in range(3)],
    )

    doc = Document(obj)
    assert doc.tags['comments'] == ['hello world', 'goodbye world']
    assert doc.tags['ratings'] == [1, 5, 4, 2]
    assert len(doc.chunks[0].chunks) == 3
    for image_doc in doc.chunks[0].chunks:
        assert image_doc.tensor.shape == (10, 10, 3)

    assert doc.is_multimodal

    expected_schema = [
        ('comments', AttributeType.ITERABLE_PRIMITIVE, 'List[str]', None),
        ('ratings', AttributeType.ITERABLE_PRIMITIVE, 'List[int]', None),
        ('images', AttributeType.ITERABLE_DOCUMENT, 'List[Image]', 0),
    ]
    _assert_doc_schema(doc, expected_schema)

    translated_obj = SocialPost(doc)
    assert translated_obj == obj


def test_iterable_nested():
    @dataclass
    class SubtitleDocument:
        text: Text
        frames: List[int]

    @dataclass
    class VideoDocument:
        frames: List[Image]
        subtitles: List[SubtitleDocument]

    obj = VideoDocument(
        frames=[np.random.rand(10, 10, 3) for _ in range(3)],
        subtitles=[
            SubtitleDocument(text='subtitle 0', frames=[0]),
            SubtitleDocument(text='subtitle 1', frames=[1, 2]),
        ],
    )

    doc = Document(obj)

    assert len(doc.chunks) == 2
    assert len(doc.chunks[0].chunks) == 3
    for image_doc in doc.chunks[0].chunks:
        assert image_doc.tensor.shape == (10, 10, 3)

    assert len(doc.chunks[1].chunks) == 2
    for i, subtitle_doc in enumerate(doc.chunks[1].chunks):
        assert subtitle_doc.chunks[0].text == f'subtitle {i}'

    assert doc.is_multimodal

    expected_schema = [
        ('frames', AttributeType.ITERABLE_DOCUMENT, 'List[Image]', 0),
        ('subtitles', AttributeType.ITERABLE_NESTED, 'List[SubtitleDocument]', 1),
    ]
    _assert_doc_schema(doc, expected_schema)

    expected_nested_schema = [
        ('text', AttributeType.DOCUMENT, 'Text', 0),
        ('frames', AttributeType.ITERABLE_PRIMITIVE, 'List[int]', None),
    ]
    for subtitle in doc.chunks[1].chunks:
        _assert_doc_schema(subtitle, expected_nested_schema)

    translated_obj = VideoDocument(doc)
    assert translated_obj == obj


def test_get_multi_modal_attribute():
    @dataclass
    class MMDocument:
        image: Image
        texts: List[Text]
        audio: Audio
        primitive: int

    mm_doc = MMDocument(
        image=np.random.rand(10, 10, 3),
        texts=['text 1', 'text 2'],
        audio=AUDIO_URI,
        primitive=1,
    )

    doc = Document(mm_doc)
    images = doc.get_multi_modal_attribute('image')
    texts = doc.get_multi_modal_attribute('texts')
    audios = doc.get_multi_modal_attribute('audio')

    assert len(images) == 1
    assert len(texts) == 2
    assert len(audios) == 1

    assert images[0].tensor.shape == (10, 10, 3)
    assert texts[0].text == 'text 1'
    assert audios[0].tensor.shape == (30833,)

    with pytest.raises(ValueError):
        doc.get_multi_modal_attribute('primitive')


@pytest.mark.parametrize(
    'text_selector',
    [
        '@.[text]',
        '@r.[text]',
        '@r. [ text]',
        '@r:.[text]',
    ],
)
@pytest.mark.parametrize(
    'audio_selector',
    ['@.[audio]', '@r.[audio]', '@r. [ audio]', '@r:.[audio]'],
)
def test_traverse_simple(text_selector, audio_selector):
    from PIL.Image import open as PIL_open

    @dataclass
    class MMDocument:
        text: Text
        audio: Audio
        image: Image

    mm_docs = DocumentArray(
        [
            Document(
                MMDocument(text=f'text {i}', audio=AUDIO_URI, image=PIL_open(IMAGE_URI))
            )
            for i in range(5)
        ]
    )

    assert len(mm_docs[text_selector]) == 5
    for i, doc in enumerate(mm_docs[text_selector]):
        assert doc.text == f'text {i}'

    assert len(mm_docs[audio_selector]) == 5
    for i, doc in enumerate(mm_docs[audio_selector]):
        assert doc.tensor.shape == (30833,)

    assert len(mm_docs['@r.[image]']) == 5
    for i, doc in enumerate(mm_docs['@r.[image]']):
        assert doc.tensor.shape == (85, 152, 3)


def test_traverse_attributes():
    @dataclass
    class MMDocument:
        attr1: Text
        attr2: Audio
        attr3: Image

    mm_docs = DocumentArray(
        [
            Document(
                MMDocument(
                    attr1='text',
                    attr2=AUDIO_URI,
                    attr3=np.random.rand(10, 10, 3),
                )
            )
            for _ in range(5)
        ]
    )

    assert len(mm_docs['@.[attr1,attr3]']) == 10
    for i, doc in enumerate(mm_docs['@.[attr1,attr3]']):
        if i % 2 == 0:
            assert doc.text == 'text'
        else:
            assert doc.tensor.shape == (10, 10, 3)


@pytest.mark.parametrize('selector', ['@r-3:.[attr]', '@r[-3:].[attr]'])
def test_traverse_slice(selector):
    @dataclass
    class MMDocument:
        attr: Text

    mm_docs = DocumentArray(
        [
            Document(
                MMDocument(
                    attr=f'text {i}',
                )
            )
            for i in range(5)
        ]
    )

    assert len(mm_docs[selector]) == 3
    for i, doc in enumerate(mm_docs[selector], start=2):
        assert doc.text == f'text {i}'


def test_traverse_iterable():
    @dataclass
    class MMDocument:
        attr1: List[Text]
        attr2: List[Audio]

    mm_da = DocumentArray(
        [
            Document(
                MMDocument(attr1=['text 1', 'text 2', 'text 3'], attr2=[AUDIO_URI] * 2)
            ),
            Document(MMDocument(attr1=['text 3', 'text 4'], attr2=[AUDIO_URI] * 3)),
        ]
    )

    assert len(mm_da['@.[attr1]']) == 5
    assert len(mm_da['@.[attr2]']) == 5

    assert len(mm_da['@.[attr1]:1']) == 2
    assert len(mm_da['@.[attr1]:1,.[attr2]-2:']) == 6

    for i, text_doc in enumerate(mm_da['@.[attr1]:2'], start=1):
        assert text_doc.text == f'text {i}'

    for i, blob_doc in enumerate(mm_da['@.[attr2]-2:'], start=1):
        assert blob_doc.tensor.shape == (30833,)


def test_traverse_chunks_attribute():
    @dataclass
    class MMDocument:
        attr: Text

    da = DocumentArray.empty(5)
    for i, d in enumerate(da):
        d.chunks.extend([Document(MMDocument(attr=f'text {i}{j}')) for j in range(5)])

    assert len(da['@r:3c:2.[attr]']) == 6
    for i, doc in enumerate(da['@r:3c:2.[attr]']):
        assert doc.text == f'text {int(i / 2)}{i % 2}'


def test_paths_separator():
    @dataclass
    class MMDocument:
        attr0: Text
        attr1: Text
        attr2: Text
        attr3: Text
        attr4: Text

    da = DocumentArray(
        [
            Document(MMDocument(**{f'attr{i}': f'text {i}' for i in range(5)}))
            for _ in range(3)
        ]
    )
    assert len(da['@r:2.[attr0,attr2,attr3],r:2.[attr1,attr4]']) == 10

    flattened = da['@r.[attr0,attr1,attr2],.[attr3,attr4]']
    for i in range(9):
        if i % 3 == 0:
            assert flattened[i].text == 'text 0'
        elif i % 3 == 1:
            assert flattened[i].text == 'text 1'
        else:
            assert flattened[i].text == 'text 2'

    for i in range(9, 15):
        if i % 2 == 1:
            assert flattened[i].text == 'text 3'
        else:
            assert flattened[i].text == 'text 4'


def test_proto_serialization():
    @dataclass
    class MMDocument:
        title: Text
        image: Image
        version: int

    obj = MMDocument(title='hello world', image=np.random.rand(10, 10, 3), version=20)
    doc = Document(obj)

    proto = doc.to_protobuf()
    assert proto._metadata is not None
    assert proto._metadata[DocumentMetadata.MULTI_MODAL_SCHEMA]

    deserialized_doc = Document.from_protobuf(proto)

    assert deserialized_doc.chunks[0].text == 'hello world'
    assert deserialized_doc.chunks[1].tensor.shape == (10, 10, 3)
    assert deserialized_doc.tags['version'] == 20

    images = deserialized_doc.get_multi_modal_attribute('image')
    titles = doc.get_multi_modal_attribute('title')

    assert images[0].tensor.shape == (10, 10, 3)
    assert titles[0].text == 'hello world'

    assert DocumentMetadata.MULTI_MODAL_SCHEMA in deserialized_doc._metadata

    expected_schema = [
        ('title', AttributeType.DOCUMENT, 'Text', 0),
        ('image', AttributeType.DOCUMENT, 'Image', 1),
        ('version', AttributeType.PRIMITIVE, 'int', None),
    ]
    _assert_doc_schema(deserialized_doc, expected_schema)

    translated_obj = MMDocument(deserialized_doc)
    assert (translated_obj.image == obj.image).all()
    assert translated_obj.title == obj.title
    assert translated_obj.version == obj.version


def test_json_type():
    @dataclass
    class MMDocument:
        attr: JSON

    inp = {'a': 123, 'b': 'abc', 'c': 1.1}
    obj = MMDocument(attr=inp)
    doc = Document(obj)

    assert doc.chunks[0].tags == inp
    translated_obj = MMDocument(doc)
    assert translated_obj == obj


def test_custom_field_type():
    from PIL.Image import Image as PILImage
    from PIL.Image import open as PIL_open

    def ndarray_serializer(value):
        return Document(blob=base64.b64encode(value))

    def ndarray_deserializer(doc: 'Document'):
        return np.frombuffer(base64.decodebytes(doc.blob), dtype=np.float64)

    def pil_image_serializer(val):
        return Document(blob=pickle.dumps(val))

    def pil_image_deserializer(doc: 'Document'):
        return pickle.loads(doc.blob)

    @dataclass
    class MMDocument:
        base64_encoded_ndarray: str = field(
            setter=ndarray_serializer, getter=ndarray_deserializer
        )
        pickled_image: PILImage = field(
            setter=pil_image_serializer, getter=pil_image_deserializer
        )

    obj = MMDocument(
        base64_encoded_ndarray=np.array([1, 2, 3], dtype=np.float64),
        pickled_image=PIL_open(IMAGE_URI),
    )

    doc = Document(obj)

    assert doc.chunks[0].blob is not None
    assert doc.chunks[1].blob is not None

    translated_obj: MMDocument = MMDocument(doc)
    assert isinstance(translated_obj.pickled_image, PILImage)
    assert isinstance(translated_obj.base64_encoded_ndarray, np.ndarray)

    assert (obj.base64_encoded_ndarray == translated_obj.base64_encoded_ndarray).all()

    assert (
        np.array(obj.pickled_image).shape
        == np.array(translated_obj.pickled_image).shape
    )


def test_invalid_type_annotations():
    @dataclass
    class MMDocument:
        attr: List

    inp = ['something']
    obj = MMDocument(attr=inp)
    with pytest.raises(Exception) as exc_info:
        Document(obj)
    assert 'Unsupported type annotation' in exc_info.value.args[0]
    assert 'Unsupported type annotation' in str(exc_info.value)


def test_not_data_class():
    class MMDocument:
        pass

    obj = MMDocument()

    with pytest.raises(Exception) as exc_info:
        Document._from_dataclass(obj)
    assert 'not a `docarray.dataclass` instance' in exc_info.value.args[0]
    assert 'not a `docarray.dataclass`' in str(exc_info.value)

    with pytest.raises(Exception) as exc_info:
        Document(obj)
        assert 'Failed to initialize' in str(exc_info.value)


def test_data_class_customized_typevar_map():
    def sette2(value):
        doc = Document(uri=value)
        doc._metadata[DocumentMetadata.IMAGE_TYPE] = 'uri'
        doc._metadata[DocumentMetadata.IMAGE_URI] = value
        doc.load_uri_to_blob()
        doc.modality = 'image'
        return doc

    type_var_m = {
        Image: lambda x: field(
            setter=sette2,
            getter=image_getter,
            _source_field=x,
        )
    }

    @dataclass(type_var_map=type_var_m)
    class MMDocument:
        image: Image
        t: Text

    d = Document(MMDocument(image=IMAGE_URI, t='hello world'))
    assert len(d.chunks) == 2
    assert d.chunks[0].blob
    assert not d.chunks[0].tensor


def test_field_fn():
    @dataclass
    class MMDoc:
        banner: List[Image] = field(default_factory=lambda: [IMAGE_URI, IMAGE_URI])

    m1 = MMDoc()
    m2 = MMDoc(Document(m1))
    assert m1 == m2


def _serialize_deserialize(doc, serialization_type):
    if serialization_type == 'protobuf':
        return Document.from_protobuf(doc.to_protobuf())
    if serialization_type == 'pickle':
        return Document.from_bytes(doc.to_bytes())
    if serialization_type == 'json':
        return Document.from_json(doc.to_json())
    if serialization_type == 'dict':
        return Document.from_dict(doc.to_dict())
    if serialization_type == 'base64':
        return Document.from_base64(doc.to_base64())
    return doc


MyText = TypeVar('MyText', bound=str)


def my_setter(value) -> 'Document':
    return Document(text=value + ' but custom!', tags={'custom': 'tag'})


def my_getter(doc: 'Document'):
    return doc.text


@dataclass
class MyMultiModalDoc:
    avatar: Image
    description: Text
    heading_list: List[Text]
    heading: MyText = field(setter=my_setter, getter=my_getter, default='')


@pytest.fixture
def mmdoc():
    return MyMultiModalDoc(
        avatar=os.path.join(cur_dir, 'toydata/test.png'),
        description='hello, world',
        heading='hello, world',
        heading_list=['hello', 'world'],
    )


@dataclass
class InnerDoc:
    avatar: Image
    description: Text
    heading: MyText = field(setter=my_setter, getter=my_getter, default='')


@dataclass
class NestedMultiModalDoc:
    other_doc: InnerDoc
    other_doc_list: List[InnerDoc]


@pytest.fixture
def nested_mmdoc():
    inner_doc = InnerDoc(
        avatar=os.path.join(cur_dir, 'toydata/test.png'),
        description='inner hello, world',
        heading='inner hello, world',
    )

    inner_doc_list = [
        InnerDoc(
            avatar=os.path.join(cur_dir, 'toydata/test.png'),
            description='inner list hello, world',
            heading=f'{i} inner list hello, world',
        )
        for i in range(3)
    ]
    return NestedMultiModalDoc(other_doc=inner_doc, other_doc_list=inner_doc_list)


@pytest.mark.parametrize(
    'serialization', [None, 'protobuf', 'pickle', 'json', 'dict', 'base64']
)
def test_multimodal_serialize_deserialize(serialization, mmdoc):
    doc = Document(mmdoc)
    assert doc._metadata
    assert doc.is_multimodal
    _metadata_before = doc._metadata
    doc = _serialize_deserialize(doc, serialization)
    assert doc._metadata
    assert doc.is_multimodal
    assert _metadata_before == doc._metadata


@pytest.mark.parametrize(
    'serialization', [None, 'protobuf', 'pickle', 'json', 'dict', 'base64']
)
def test_get_multimodal(serialization, mmdoc):
    d = Document(mmdoc)
    if serialization:
        d = _serialize_deserialize(d, serialization)

    assert isinstance(d.description, Document)
    assert d.description.content == 'hello, world'
    assert isinstance(d.heading, Document)
    assert d.heading.content == 'hello, world but custom!'
    assert d.heading.tags == {'custom': 'tag'}
    assert isinstance(d.heading_list, DocumentArray)
    assert d.heading_list.texts == ['hello', 'world']


@pytest.mark.parametrize(
    'serialization', [None, 'protobuf', 'pickle', 'json', 'dict', 'base64']
)
def test_set_multimodal(serialization, mmdoc):
    d = Document(mmdoc)
    if serialization:
        d = _serialize_deserialize(d, serialization)

    d.description = Document(text='hello, beautiful world')
    d.heading.text = 'hello, world but beautifully custom!'
    d.heading_list = DocumentArray(
        [Document(text=t) for t in ['hello', 'beautiful', 'world']]
    )

    assert isinstance(d.description, Document)
    assert d.description.content == 'hello, beautiful world'
    assert d.description in d.chunks
    assert DocumentArray(d)['@.[description]'][0] == d.description

    assert isinstance(d.heading, Document)
    assert d.heading.content == 'hello, world but beautifully custom!'
    assert d.heading.tags == {'custom': 'tag'}
    assert d.heading in d.chunks
    assert DocumentArray(d)['@.[heading]'][0] == d.heading

    assert isinstance(d.heading_list, DocumentArray)
    assert d.heading_list.texts == ['hello', 'beautiful', 'world']
    heading_list_in_chunks = False
    for c in d.chunks:
        if c.chunks == d.heading_list:
            heading_list_in_chunks = True
    assert heading_list_in_chunks
    for d1, d2 in zip(DocumentArray(d)['@.[heading_list]'], d.heading_list):
        assert d1 == d2


@pytest.mark.parametrize(
    'serialization', [None, 'protobuf', 'pickle', 'json', 'dict', 'base64']
)
def test_access_multimodal_nested(serialization, nested_mmdoc):
    d = Document(nested_mmdoc)
    if serialization:
        d = _serialize_deserialize(d, serialization)

    assert isinstance(d.other_doc, Document)
    assert d.other_doc.is_multimodal
    assert d.other_doc.heading.content == 'inner hello, world but custom!'
    assert isinstance(d.other_doc_list, DocumentArray)
    assert isinstance(d.other_doc_list[0], Document)
    assert (
        d.other_doc_list[1].heading.content == '1 inner list hello, world but custom!'
    )


@pytest.mark.parametrize(
    'serialization', [None, 'protobuf', 'pickle', 'json', 'dict', 'base64']
)
def test_set_multimodal_nested(serialization, nested_mmdoc):
    d = Document(nested_mmdoc)
    if serialization:
        d = _serialize_deserialize(d, serialization)

    new_inner_doc = Document(text='new text inner doc')
    new_inner_list_doc = Document(text='1 new text list')
    d.other_doc.heading = new_inner_doc
    d.other_doc_list[0].heading.text = '0 new text list'
    d.other_doc_list[1].heading = new_inner_list_doc

    assert d.other_doc.heading.text == 'new text inner doc'
    assert d.other_doc.heading in d.other_doc.chunks
    heading_in_chunk_of_chunks = False
    for c in d.chunks:
        if c.chunks:
            for cc in c.chunks:
                if cc == d.other_doc.heading:
                    heading_in_chunk_of_chunks = True
    assert heading_in_chunk_of_chunks
    assert DocumentArray(d.other_doc)['@.[heading]'][0] == new_inner_doc

    assert d.other_doc_list[0].heading.text == '0 new text list'
    assert '0 new text list' in d.other_doc_list['@.[heading]'][:, 'text']

    assert d.other_doc_list[1].heading.text == '1 new text list'
    assert new_inner_list_doc in d.other_doc_list['@.[heading]']


def test_initialize_document_with_dataclass_and_additional_text_attr():
    @dataclass
    class MyDoc:
        chunk_text: Text

    d = Document(MyDoc(chunk_text='chunk level text'), text='top level text')

    assert d.text == 'top level text'
    assert d.chunk_text.text == 'chunk level text'


def test_initialize_document_with_dataclass_and_additional_unknown_attributes():
    @dataclass
    class MyDoc:
        chunk_text: Text

    d = Document(
        MyDoc(chunk_text='chunk level text'),
        hello='top level text',
    )

    assert d.tags['hello'] == 'top level text'
    assert d.chunk_text.text == 'chunk level text'


def test_doc_with_dataclass_with_str_attr_and_additional_unknown_attribute():
    @dataclass
    class MyDoc:
        name_mydoc: str

    d = Document(MyDoc(name_mydoc='mydoc'), name_doc='doc')

    assert d.tags['name_mydoc'] == 'mydoc'
    assert d.tags['name_doc'] == 'doc'


def test_doc_with_dataclass_with_str_attr_and_additional_tags_arg():
    @dataclass
    class MyDoc:
        name_mydoc: str

    d = Document(MyDoc(name_mydoc='mydoc'), tags={'name_doc': 'doc'})

    assert d.tags['name_mydoc'] == 'mydoc'
    assert d.tags['name_doc'] == 'doc'


def test_doc_with_dataclass_with_str_and_additional_tags_arg_and_unknown_attribute():
    @dataclass
    class MyDoc:
        name_mydoc: str

    d = Document(
        MyDoc(name_mydoc='mydoc'), tags={'name_doc': 'doc'}, something_else='hello'
    )

    assert d.tags['name_mydoc'] == 'mydoc'
    assert d.tags['name_doc'] == 'doc'
    assert d.tags['something_else'] == 'hello'


def test_doc_with_dataclass_with_str_attr_and_additional_unknown_attr_with_same_name():
    @dataclass
    class MyDoc:
        name: str

    d = Document(MyDoc(name='mydoc'), name='doc')

    assert d.tags['name'] == 'doc'


def test_empty_list_dataclass():
    @dataclass()
    class A:
        text: List[Text]

    doc = Document(A(text=[]))


def test_doc_with_dataclass_with_list_of_length_one():
    @dataclass
    class MyDoc:
        title: Text
        images: List[Image]

    doc = Document(MyDoc(title='doc 1', images=[IMAGE_URI]))
    assert type(doc.images) == ChunkArray
    assert len(doc.images) == 1


def test_doc_with_dataclass_without_list():
    @dataclass
    class MyDoc:
        title: Text
        image: Image

    doc = Document(MyDoc(title='doc 1', image=IMAGE_URI))
    assert type(doc.image) == Document
