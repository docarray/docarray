from pathlib import Path

import numpy as np
import pytest
from pydantic.tools import parse_obj_as

from docarray.typing import MeshUrl

REPO_ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent.absolute()
TOYDATA_DIR = REPO_ROOT_DIR / 'tests' / 'toydata'

MESH_FILES = {
    'obj': str(TOYDATA_DIR / 'tetrahedron.obj'),
    'glb': str(TOYDATA_DIR / 'test.glb'),
    'ply': str(TOYDATA_DIR / 'cube.ply'),
}
REMOTE_OBJ_FILE = 'https://people.sc.fsu.edu/~jburkardt/data/obj/al.obj'


@pytest.mark.slow
@pytest.mark.internet
def test_image_url():
    uri = parse_obj_as(MeshUrl, REMOTE_OBJ_FILE)

    vertices, faces = uri.load()

    assert isinstance(vertices, np.ndarray)
    assert isinstance(faces, np.ndarray)


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
    url = parse_obj_as(MeshUrl, file_path)
    vertices, faces = url.load()
    assert isinstance(vertices, np.ndarray)
    assert isinstance(faces, np.ndarray)
