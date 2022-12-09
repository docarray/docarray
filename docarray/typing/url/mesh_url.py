from typing import Tuple, TypeVar

import numpy as np

from docarray.typing.url.any_url import AnyUrl
from docarray.typing.url.helper_3d_data import load_trimesh_instance

T = TypeVar('T', bound='MeshUrl')


class MeshUrl(AnyUrl):
    """
    URL to a .obj, .glb, or .ply file.
    Can be remote (web) URL, or a local file path.
    """

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the data from the url into a tuple of two numpy.ndarrays containing
        vertices and faces information.

        EXAMPLE USAGE

        .. code-block:: python

            from docarray import Document
            import numpy as np

            from docarray.typing.url.mesh_url import MeshUrl


            class MyDoc(Document):
                mesh_url: MeshUrl


            doc = MyDoc(mesh_url="toydata/tetrahedron.obj")

            vertices, faces = doc.mesh_url.load()
            assert isinstance(vertices, np.ndarray)
            assert isinstance(faces, np.ndarray)

        :return: np.ndarray representing the image as RGB values
        """

        mesh = load_trimesh_instance(uri=self, force='mesh')

        vertices = mesh.vertices.view(np.ndarray)
        faces = mesh.faces.view(np.ndarray)

        return vertices, faces
