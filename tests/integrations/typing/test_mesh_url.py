from docarray import BaseDoc
from docarray.typing import Mesh3DUrl


def test_set_mesh_url():
    class MyDocument(BaseDoc):
        mesh_url: Mesh3DUrl

    d = MyDocument(mesh_url="https://jina.ai/mesh.obj")

    assert isinstance(d.mesh_url, Mesh3DUrl)
    assert d.mesh_url == "https://jina.ai/mesh.obj"
