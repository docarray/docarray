from typing import TYPE_CHECKING, Any, Tuple, Type, TypeVar, Union

import numpy as np

from docarray.typing.url.any_url import AnyUrl
from docarray.typing.url.helper_3d_data import MESH_FILE_FORMATS, _load_trimesh_instance

if TYPE_CHECKING:
    from pydantic import BaseConfig
    from pydantic.fields import ModelField

T = TypeVar('T', bound='MeshUrl')


class MeshUrl(AnyUrl):
    """
    URL to a .obj, .glb, or .ply file.
    Can be remote (web) URL, or a local file path.
    """

    @classmethod
    def validate(
        cls: Type[T],
        value: Union[T, np.ndarray, Any],
        field: 'ModelField',
        config: 'BaseConfig',
    ) -> T:
        url = super().validate(value, field, config)  # basic url validation
        has_image_extension = any(url.endswith(ext) for ext in MESH_FILE_FORMATS)
        if not has_image_extension:
            raise ValueError(
                f'Mesh URL must have one of the following extensions:'
                f'{MESH_FILE_FORMATS}'
            )
        return cls(str(url), scheme=None)

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

        :return: tuple of two np.ndarrays representing the mesh's vertices and faces
        """

        mesh = _load_trimesh_instance(uri=self, force='mesh')

        vertices = mesh.vertices.view(np.ndarray)
        faces = mesh.faces.view(np.ndarray)

        return vertices, faces
