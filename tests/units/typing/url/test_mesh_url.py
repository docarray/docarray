import numpy as np
import pytest
from pydantic.tools import parse_obj_as, schema_json_of

from docarray.document.io.json import orjson_dumps_and_decode
from docarray.typing import Mesh3DUrl
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
    vertices, faces = url.load()

    assert isinstance(vertices, np.ndarray)
    assert isinstance(faces, np.ndarray)
    assert vertices.shape[1] == 3
    assert faces.shape[1] == 3


def test_json_schema():
    schema_json_of(Mesh3DUrl)


def test_dump_json():
    url = parse_obj_as(Mesh3DUrl, REMOTE_OBJ_FILE)
    orjson_dumps_and_decode(url)


@pytest.mark.parametrize(
    'file_format,path_to_file',
    [
        ('obj', MESH_FILES['obj']),
        ('glb', MESH_FILES['glb']),
        ('ply', MESH_FILES['ply']),
        ('obj', REMOTE_OBJ_FILE),
        ('illegal', 'illegal'),
        ('illegal', 'https://www.google.com'),
        ('illegal', 'my/local/text/file.txt'),
        ('illegal', 'my/local/text/file.png'),
    ],
)
def test_validation(file_format, path_to_file):
    if file_format == 'illegal':
        with pytest.raises(ValueError, match='Mesh3DUrl'):
            parse_obj_as(Mesh3DUrl, path_to_file)
    else:
        url = parse_obj_as(Mesh3DUrl, path_to_file)
        assert isinstance(url, Mesh3DUrl)
        assert isinstance(url, str)


def test_proto_mesh_url():
    uri = parse_obj_as(Mesh3DUrl, REMOTE_OBJ_FILE)
    uri._to_node_protobuf()
