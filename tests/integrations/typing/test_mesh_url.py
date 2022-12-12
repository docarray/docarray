from docarray import Document
from docarray.typing import MeshUrl


def test_set_mesh_url():
    class MyDocument(Document):
        mesh_url: MeshUrl

    d = MyDocument(mesh_url="https://jina.ai/mesh.obj")

    assert isinstance(d.mesh_url, MeshUrl)
    assert d.mesh_url == "https://jina.ai/mesh.obj"
