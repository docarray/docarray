import numpy as np
import pytest

from docarray import Mesh

REMOTE_OBJ_FILE = 'https://people.sc.fsu.edu/~jburkardt/data/obj/al.obj'


@pytest.mark.slow
@pytest.mark.internet
def test_mesh():

    mesh = Mesh(url=REMOTE_OBJ_FILE)

    mesh.vertices, mesh.faces = mesh.url.load()

    assert isinstance(mesh.vertices, np.ndarray)
    assert isinstance(mesh.faces, np.ndarray)
