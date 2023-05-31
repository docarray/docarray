import os

import numpy as np
import pytest
from pydantic.tools import parse_obj_as, schema_json_of

from docarray.base_doc.io.json import orjson_dumps
from docarray.typing import NdArray, PointCloud3DUrl
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
    n_samples = 100
    url = parse_obj_as(PointCloud3DUrl, file_path)
    tensors = url.load(samples=n_samples)

    assert isinstance(tensors.points, np.ndarray)
    assert isinstance(tensors.points, NdArray)
    assert tensors.points.shape == (n_samples, 3)


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
def test_load_with_multiple_geometries_true(file_format, file_path):
    n_samples = 100
    url = parse_obj_as(PointCloud3DUrl, file_path)
    tensors = url.load(samples=n_samples, multiple_geometries=True)

    assert isinstance(tensors.points, np.ndarray)
    assert len(tensors.points.shape) == 3
    assert tensors.points.shape[1:] == (100, 3)


def test_json_schema():
    schema_json_of(PointCloud3DUrl)


def test_dump_json():
    url = parse_obj_as(PointCloud3DUrl, REMOTE_OBJ_FILE)
    orjson_dumps(url)


@pytest.mark.parametrize(
    'path_to_file',
    [*MESH_FILES.values(), REMOTE_OBJ_FILE],
)
def test_validation(path_to_file):
    url = parse_obj_as(PointCloud3DUrl, path_to_file)
    assert isinstance(url, PointCloud3DUrl)
    assert isinstance(url, str)


@pytest.mark.proto
def test_proto_point_cloud_url():
    uri = parse_obj_as(PointCloud3DUrl, REMOTE_OBJ_FILE)
    uri._to_node_protobuf()


@pytest.mark.parametrize(
    'file_type, file_source',
    [
        ('application', MESH_FILES['obj']),
        ('application', MESH_FILES['glb']),
        ('application', MESH_FILES['ply']),
        ('application', REMOTE_OBJ_FILE),
        ('audio', os.path.join(TOYDATA_DIR, 'hello.aac')),
        ('audio', os.path.join(TOYDATA_DIR, 'hello.mp3')),
        ('audio', os.path.join(TOYDATA_DIR, 'hello.ogg')),
        ('video', os.path.join(TOYDATA_DIR, 'mov_bbb.mp4')),
        ('image', os.path.join(TOYDATA_DIR, 'test.png')),
        ('text', os.path.join(TOYDATA_DIR, 'test' 'test.html')),
        ('text', os.path.join(TOYDATA_DIR, 'test' 'test.md')),
        ('text', os.path.join(TOYDATA_DIR, 'penal_colony.txt')),
    ],
)
def test_file_validation(file_type, file_source):
    if file_type != PointCloud3DUrl.mime_type():
        with pytest.raises(ValueError):
            parse_obj_as(PointCloud3DUrl, file_source)
    else:
        parse_obj_as(PointCloud3DUrl, file_source)
