import numpy as np
import pytest
from pydantic import parse_obj_as

from docarray.base_document.doc import BaseDoc
from docarray.documents import Mesh3D
from tests import TOYDATA_DIR

LOCAL_OBJ_FILE = str(TOYDATA_DIR / 'tetrahedron.obj')
REMOTE_OBJ_FILE = 'https://people.sc.fsu.edu/~jburkardt/data/obj/al.obj'


@pytest.mark.slow
@pytest.mark.internet
@pytest.mark.parametrize('file_url', [LOCAL_OBJ_FILE, REMOTE_OBJ_FILE])
def test_mesh(file_url):

    mesh = Mesh3D(url=file_url)

    mesh.tensors = mesh.url.load()

    assert isinstance(mesh.tensors.vertices, np.ndarray)
    assert isinstance(mesh.tensors.faces, np.ndarray)


def test_str_init():
    t = parse_obj_as(Mesh3D, 'http://hello.ply')
    assert t.url == 'http://hello.ply'


def test_doc():
    class MyDoc(BaseDoc):
        mesh1: Mesh3D
        mesh2: Mesh3D

    doc = MyDoc(mesh1='http://hello.ply', mesh2=Mesh3D(url='http://hello.ply'))

    assert doc.mesh1.url == 'http://hello.ply'
    assert doc.mesh2.url == 'http://hello.ply'
