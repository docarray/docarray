import numpy as np
import pytest
from pydantic.tools import parse_obj_as, schema_json_of

from docarray.document_base.io.json import orjson_dumps
from docarray.typing import PointCloud3DUrl
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
    point_cloud = url.load(samples=n_samples)

    assert isinstance(point_cloud, np.ndarray)
    assert point_cloud.shape == (n_samples, 3)


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
    point_cloud = url.load(samples=n_samples, multiple_geometries=True)

    assert isinstance(point_cloud, np.ndarray)
    assert len(point_cloud.shape) == 3
    assert point_cloud.shape[1:] == (100, 3)


def test_json_schema():
    schema_json_of(PointCloud3DUrl)


def test_dump_json():
    url = parse_obj_as(PointCloud3DUrl, REMOTE_OBJ_FILE)
    orjson_dumps(url)


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
        with pytest.raises(ValueError, match='PointCloud3DUrl'):
            parse_obj_as(PointCloud3DUrl, path_to_file)
    else:
        url = parse_obj_as(PointCloud3DUrl, path_to_file)
        assert isinstance(url, PointCloud3DUrl)
        assert isinstance(url, str)


def test_proto_point_cloud_url():
    uri = parse_obj_as(PointCloud3DUrl, REMOTE_OBJ_FILE)
    uri._to_node_protobuf()
