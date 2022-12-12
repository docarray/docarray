from typing import Optional

from docarray.document import BaseDocument
from docarray.typing import Embedding, MeshUrl, Tensor


class Mesh(BaseDocument):
    """
    Document for handling meshes for 3D data representation.
    It can contain an MeshUrl (`Mesh.url`), a Tensor (`Mesh.tensor`),
    and an Embedding (`Mesh.embedding`).

    EXAMPLE USAGE:

    You can use this Document directly:

    .. code-block:: python

        from docarray import Mesh

        # use it directly
        mesh = Mesh(url='https://people.sc.fsu.edu/~jburkardt/data/obj/al.obj')
        mesh.tensor = mesh.url.load()
        model = MyEmbeddingModel()
        mesh.embedding = model(mesh.tensor)

    You can extend this Document:

    .. code-block:: python

        from docarray import Mesh
        from docarray.typing import Embedding
        from typing import Optional

        # extend it
        class MyMesh(Mesh):
            second_embedding: Optional[Embedding]


        mesh = MyMesh(url='https://people.sc.fsu.edu/~jburkardt/data/obj/al.obj')
        mesh.tensor = mesh.url.load()
        model = MyEmbeddingModel()
        mesh.embedding = model(mesh.tensor)
        mesh.second_embedding = model(mesh.tensor)


    You can use this Document for composition:

    .. code-block:: python

        from docarray import Document, Mesh, Text

        # compose it
        class MultiModalDoc(Document):
            mesh: Mesh
            text: Text


        mmdoc = MultiModalDoc(
            mesh=Mesh(url="https://people.sc.fsu.edu/~jburkardt/data/obj/al.obj"),
            text=Text(text="hello world, how are you doing?"),
        )
        mmdoc.mesh.tensor = mmdoc.mesh.url.load()
    """

    url: Optional[MeshUrl]
    vertices: Optional[Tensor]
    faces: Optional[Tensor]
    embedding: Optional[Embedding]
