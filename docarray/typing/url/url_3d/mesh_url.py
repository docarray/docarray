from typing import TYPE_CHECKING, TypeVar

import numpy as np
from pydantic import parse_obj_as

from docarray.typing.proto_register import _register_proto
from docarray.typing.tensor.ndarray import NdArray
from docarray.typing.url.url_3d.url_3d import Url3D

if TYPE_CHECKING:
    from docarray.documents.mesh.vertices_and_faces import VerticesAndFaces

T = TypeVar('T', bound='Mesh3DUrl')


@_register_proto(proto_type_name='mesh_url')
class Mesh3DUrl(Url3D):
    """
    URL to a .obj, .glb, or .ply file containing 3D mesh information.
    Can be remote (web) URL, or a local file path.
    """

    def load(self: T) -> 'VerticesAndFaces':
        """
        Load the data from the url into a VerticesAndFaces object containing
        vertices and faces information.

        EXAMPLE USAGE

        .. code-block:: python

            from docarray import BaseDocument
            import numpy as np

            from docarray.typing import Mesh3DUrl, NdArray


            class MyDoc(BaseDocument):
                mesh_url: Mesh3DUrl


            doc = MyDoc(mesh_url="toydata/tetrahedron.obj")

            tensors = doc.mesh_url.load()
            assert isinstance(tensors.vertices, NdArray)
            assert isinstance(tensors.faces, NdArray)


        :return: VerticesAndFaces object containing vertices and faces information.
        """
        from docarray.documents.mesh.vertices_and_faces import VerticesAndFaces

        mesh = self._load_trimesh_instance(force='mesh')

        vertices = parse_obj_as(NdArray, mesh.vertices.view(np.ndarray))
        faces = parse_obj_as(NdArray, mesh.faces.view(np.ndarray))

        return VerticesAndFaces(vertices=vertices, faces=faces)

    def display(self) -> None:
        """
        Plot mesh from url.
        This loads the Trimesh instance of the 3D mesh, and then displays it.
        """
        from IPython.display import display

        mesh = self._load_trimesh_instance()
        display(mesh.show())
