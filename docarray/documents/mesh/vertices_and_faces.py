from typing import TYPE_CHECKING, Any, Type, TypeVar, Union

from docarray.base_doc import BaseDoc
from docarray.typing.tensor.tensor import AnyTensor
from docarray.utils.misc import import_library

T = TypeVar('T', bound='VerticesAndFaces')


class VerticesAndFaces(BaseDoc):
    """
    Document for handling 3D mesh tensor data.

    A VerticesAndFaces Document can contain an AnyTensor containing the vertices
    information (`VerticesAndFaces.vertices`), and an AnyTensor containing the faces
    information (`VerticesAndFaces.faces`).
    """

    vertices: AnyTensor
    faces: AnyTensor

    @classmethod
    def validate(
        cls: Type[T],
        value: Union[str, Any],
    ) -> T:
        return super().validate(value)

    def display(self) -> None:
        """
        Plot mesh consisting of vertices and faces.
        To use this you need to install trimesh[easy]: `pip install 'trimesh[easy]'`.
        """
        if TYPE_CHECKING:
            import trimesh
        else:
            trimesh = import_library('trimesh', raise_error=True)

        from IPython.display import display

        if self.vertices is None or self.faces is None:
            raise ValueError(
                'Can\'t display mesh from tensors when the vertices and/or faces '
                'are None.'
            )

        mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces)
        display(mesh.show())
