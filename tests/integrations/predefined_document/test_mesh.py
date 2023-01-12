import numpy as np
import pytest

from docarray.documents import Mesh3D
from tests import TOYDATA_DIR

LOCAL_OBJ_FILE = str(TOYDATA_DIR / 'tetrahedron.obj')
REMOTE_OBJ_FILE = 'https://people.sc.fsu.edu/~jburkardt/data/obj/al.obj'


@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize('file_url', [LOCAL_OBJ_FILE, REMOTE_OBJ_FILE])
def test_mesh(file_url):

    mesh = Mesh3D(url=file_url)

    mesh.vertices, mesh.faces = mesh.url.load()

    assert isinstance(mesh.vertices, np.ndarray)
    assert isinstance(mesh.faces, np.ndarray)
