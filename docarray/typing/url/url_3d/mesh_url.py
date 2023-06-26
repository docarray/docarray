from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypeVar

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
    URL to a file containing 3D mesh information.
    Can be remote (web) URL, or a local file path.
    """

    @classmethod
    def extra_extensions(cls) -> List[str]:
        # return list of allowed extensions to be used for mesh if mimetypes fail to detect
        # generated with the help of chatGPT and definitely this list is not exhaustive
        # bit hacky because of black formatting, making it a long vertical list
        list_a = ['3ds', '3mf', 'ac', 'ac3d', 'amf', 'assimp', 'bvh', 'cob', 'collada']
        list_b = ['ctm', 'dxf', 'e57', 'fbx', 'gltf', 'glb', 'ifc', 'lwo', 'lws', 'lxo']
        list_c = ['md2', 'md3', 'md5', 'mdc', 'm3d', 'mdl', 'ms3d', 'nff', 'obj', 'off']
        list_d = ['pcd', 'pod', 'pmd', 'pmx', 'ply', 'q3o', 'q3s', 'raw', 'sib', 'smd']
        list_e = ['stl', 'ter' 'terragen', 'vtk', 'vrml', 'x3d', 'xaml', 'xgl', 'xml']
        list_f = ['xyz', 'zgl', 'vta']
        return list_a + list_b + list_c + list_d + list_e + list_f

    def load(
        self: T,
        skip_materials: bool = True,
        trimesh_args: Optional[Dict[str, Any]] = None,
    ) -> 'VerticesAndFaces':
        """
        Load the data from the url into a [`VerticesAndFaces`][docarray.documents.VerticesAndFaces]
        object containing vertices and faces information.

        ---

        ```python
        from docarray import BaseDoc

        from docarray.typing import Mesh3DUrl, NdArray


        class MyDoc(BaseDoc):
            mesh_url: Mesh3DUrl


        doc = MyDoc(mesh_url="https://people.sc.fsu.edu/~jburkardt/data/obj/al.obj")

        tensors = doc.mesh_url.load()
        assert isinstance(tensors.vertices, NdArray)
        assert isinstance(tensors.faces, NdArray)
        ```


        :param skip_materials: Skip materials if True, else skip.
        :param trimesh_args: dictionary of additional arguments for `trimesh.load()`
            or `trimesh.load_remote()`.
        :return: VerticesAndFaces object containing vertices and faces information.
        """
        from docarray.documents.mesh.vertices_and_faces import VerticesAndFaces

        if not trimesh_args:
            trimesh_args = {}
        mesh = self._load_trimesh_instance(
            force='mesh', skip_materials=skip_materials, **trimesh_args
        )

        vertices = parse_obj_as(NdArray, mesh.vertices.view(np.ndarray))
        faces = parse_obj_as(NdArray, mesh.faces.view(np.ndarray))

        return VerticesAndFaces(vertices=vertices, faces=faces)

    def display(self) -> None:
        """
        Plot mesh from url.
        This loads the Trimesh instance of the 3D mesh, and then displays it.
        """
        from IPython.display import display

        mesh = self._load_trimesh_instance(skip_materials=False)
        display(mesh.show())
