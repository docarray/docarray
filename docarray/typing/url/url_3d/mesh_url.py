from typing import TYPE_CHECKING, Tuple, TypeVar

import numpy as np

from docarray.typing.url.url_3d.url_3d import Url3D

if TYPE_CHECKING:
    from docarray.proto import NodeProto

T = TypeVar('T', bound='Mesh3DUrl')


class Mesh3DUrl(Url3D):
    """
    URL to a .obj, .glb, or .ply file containing 3D mesh information.
    Can be remote (web) URL, or a local file path.
    """
    _proto_type_name = 'mesh_url'

    def load(self: T) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the data from the url into a tuple of two numpy.ndarrays containing
        vertices and faces information.

        EXAMPLE USAGE

        .. code-block:: python

            from docarray import BaseDocument
            import numpy as np

            from docarray.typing import Mesh3DUrl


            class MyDoc(BaseDocument):
                mesh_url: Mesh3DUrl


            doc = MyDoc(mesh_url="toydata/tetrahedron.obj")

            vertices, faces = doc.mesh_url.load()
            assert isinstance(vertices, np.ndarray)
            assert isinstance(faces, np.ndarray)

        :return: tuple of two np.ndarrays representing the mesh's vertices and faces
        """

        mesh = self._load_trimesh_instance(force='mesh')

        vertices = mesh.vertices.view(np.ndarray)
        faces = mesh.faces.view(np.ndarray)

        return vertices, faces
