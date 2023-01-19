from typing import TYPE_CHECKING, NamedTuple, TypeVar

import numpy as np
from pydantic import parse_obj_as

from docarray.typing import NdArray
from docarray.typing.url.url_3d.url_3d import Url3D

if TYPE_CHECKING:
    from docarray.proto import NodeProto

T = TypeVar('T', bound='Mesh3DUrl')


class Mesh3DLoadResult(NamedTuple):
    vertices: NdArray
    faces: NdArray


class Mesh3DUrl(Url3D):
    """
    URL to a .obj, .glb, or .ply file containing 3D mesh information.
    Can be remote (web) URL, or a local file path.
    """

    def _to_node_protobuf(self: T) -> 'NodeProto':
        """Convert Document into a NodeProto protobuf message. This function should
        be called when the Document is nested into another Document that needs to
        be converted into a protobuf

        :return: the nested item protobuf message
        """
        from docarray.proto import NodeProto

        return NodeProto(mesh_url=str(self))

    def load(self: T) -> Mesh3DLoadResult:
        """
        Load the data from the url into a named tuple of two NdArrays containing
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

        :return: named tuple of two NdArrays representing the mesh's vertices and faces
        """

        mesh = self._load_trimesh_instance(force='mesh')

        vertices = parse_obj_as(NdArray, mesh.vertices.view(np.ndarray))
        faces = parse_obj_as(NdArray, mesh.faces.view(np.ndarray))

        return Mesh3DLoadResult(vertices=vertices, faces=faces)
