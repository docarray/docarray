from typing import Optional

from docarray.document import BaseDocument
from docarray.typing import AnyTensor, Embedding, Mesh3DUrl


class Mesh3D(BaseDocument):
    """
    Document for handling meshes for 3D data representation.

    A mesh is a representation for 3D data and contains vertices and faces information.
    Vertices are points in a 3D space, represented as a tensor of shape (n_points, 3).
    Faces are triangular surfaces that can be defined by three points in 3D space,
    corresponding to the three vertices of a triangle. Faces can be represented as a
    tensor of shape (n_faces, 3). Each number in that tensor refers to an index of a
    vertex in the tensor of vertices.

    The Mesh3D Document can contain an Mesh3DUrl (`Mesh3D.url`), a AnyTensor of vertices
    (`Mesh3D.vertices`), a AnyTensor of faces (`Mesh3D.faces`) and an Embedding
    (`Mesh3D.embedding`).

    EXAMPLE USAGE:

    You can use this Document directly:

    .. code-block:: python

        from docarray import Mesh3D

        # use it directly
        mesh = Mesh3D(url='https://people.sc.fsu.edu/~jburkardt/data/obj/al.obj')
        mesh.vertices, mesh.faces = mesh.url.load()
        model = MyEmbeddingModel()
        mesh.embedding = model(mesh.vertices)

    You can extend this Document:

    .. code-block:: python

        from docarray import Mesh3D
        from docarray.typing import Embedding
        from typing import Optional

        # extend it
        class MyMesh3D(Mesh3D):
            name: Optional[Text]


        mesh = MyMesh3D(url='https://people.sc.fsu.edu/~jburkardt/data/obj/al.obj')
        mesh.vertices, mesh.faces = mesh.url.load()
        model = MyEmbeddingModel()
        mesh.embedding = model(mesh.vertices)
        mesh.name = 'my first mesh'


    You can use this Document for composition:

    .. code-block:: python

        from docarray import Document, Mesh3D, Text

        # compose it
        class MultiModalDoc(Document):
            mesh: Mesh3D
            text: Text


        mmdoc = MultiModalDoc(
            mesh=Mesh3D(url='https://people.sc.fsu.edu/~jburkardt/data/obj/al.obj'),
            text=Text(text='hello world, how are you doing?'),
        )
        mmdoc.mesh.vertices, mmdoc.mesh.faces = mmdoc.mesh.url.load()
    """

    url: Optional[Mesh3DUrl]
    vertices: Optional[AnyTensor]
    faces: Optional[AnyTensor]
    embedding: Optional[Embedding]
