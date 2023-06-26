import os

import numpy as np
import pytest
from pydantic.tools import parse_obj_as, schema_json_of

from docarray.base_doc.io.json import orjson_dumps
from docarray.typing import Mesh3DUrl, NdArray
from docarray.typing.url.mimetypes import (
    OBJ_MIMETYPE,
    AUDIO_MIMETYPE,
    VIDEO_MIMETYPE,
    IMAGE_MIMETYPE,
    TEXT_MIMETYPE,
)
from tests import TOYDATA_DIR

MESH_FILES = {
    'obj': str(TOYDATA_DIR / 'tetrahedron.obj'),
    'glb': str(TOYDATA_DIR / 'test.glb'),
    'ply': str(TOYDATA_DIR / 'cube.ply'),
}
REMOTE_OBJ_FILE = 'https://people.sc.fsu.edu/~jburkardt/data/obj/al.obj'


@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize(
    'file_format, file_path',
    [
        ('obj', MESH_FILES['obj']),
        ('glb', MESH_FILES['glb']),
        ('ply', MESH_FILES['ply']),
        ('remote-obj', REMOTE_OBJ_FILE),
    ],
)
def test_load(file_format, file_path):
    url = parse_obj_as(Mesh3DUrl, file_path)
    tensors = url.load()

    assert isinstance(tensors.vertices, np.ndarray)
    assert isinstance(tensors.vertices, NdArray)
    assert isinstance(tensors.faces, np.ndarray)
    assert isinstance(tensors.faces, NdArray)
    assert tensors.vertices.shape[1] == 3
    assert tensors.faces.shape[1] == 3


@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize(
    'file_path',
    [*MESH_FILES.values(), REMOTE_OBJ_FILE],
)
@pytest.mark.parametrize('field', ['vertices', 'faces'])
def test_load_one_of_fields(file_path, field):
    url = parse_obj_as(Mesh3DUrl, file_path)
    field = getattr(url.load(), field)

    assert isinstance(field, np.ndarray)
    assert isinstance(field, NdArray)


def test_json_schema():
    schema_json_of(Mesh3DUrl)


def test_dump_json():
    url = parse_obj_as(Mesh3DUrl, REMOTE_OBJ_FILE)
    orjson_dumps(url)


@pytest.mark.parametrize(
    'path_to_file',
    [*MESH_FILES.values(), REMOTE_OBJ_FILE],
)
def test_validation(path_to_file):
    url = parse_obj_as(Mesh3DUrl, path_to_file)
    assert isinstance(url, Mesh3DUrl)
    assert isinstance(url, str)


@pytest.mark.proto
def test_proto_mesh_url():
    uri = parse_obj_as(Mesh3DUrl, REMOTE_OBJ_FILE)
    uri._to_node_protobuf()


@pytest.mark.parametrize(
    'file_type, file_source',
    [
        (OBJ_MIMETYPE, MESH_FILES['obj']),
        (OBJ_MIMETYPE, MESH_FILES['glb']),
        (OBJ_MIMETYPE, MESH_FILES['ply']),
        (OBJ_MIMETYPE, REMOTE_OBJ_FILE),
        (AUDIO_MIMETYPE, os.path.join(TOYDATA_DIR, 'hello.aac')),
        (AUDIO_MIMETYPE, os.path.join(TOYDATA_DIR, 'hello.mp3')),
        (AUDIO_MIMETYPE, os.path.join(TOYDATA_DIR, 'hello.ogg')),
        (VIDEO_MIMETYPE, os.path.join(TOYDATA_DIR, 'mov_bbb.mp4')),
        (IMAGE_MIMETYPE, os.path.join(TOYDATA_DIR, 'test.png')),
        (TEXT_MIMETYPE, os.path.join(TOYDATA_DIR, 'test' 'test.html')),
        (TEXT_MIMETYPE, os.path.join(TOYDATA_DIR, 'test' 'test.md')),
        (TEXT_MIMETYPE, os.path.join(TOYDATA_DIR, 'penal_colony.txt')),
    ],
)
def test_file_validation(file_type, file_source):
    if file_type != Mesh3DUrl.mime_type():
        print('1')
        with pytest.raises(ValueError):
            parse_obj_as(Mesh3DUrl, file_source)
    else:
        print('2')
        parse_obj_as(Mesh3DUrl, file_source)
