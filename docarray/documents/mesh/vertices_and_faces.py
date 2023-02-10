from typing import Any, Optional, Type, TypeVar, Union

from docarray.base_document import BaseDocument
from docarray.typing.tensor.tensor import AnyTensor

T = TypeVar('T', bound='VerticesAndFaces')


class VerticesAndFaces(BaseDocument):
    """
    Document for handling 3D mesh tensor data.

    A VerticesAndFaces Document can contain an AnyTensor containing the vertices
    information (`VerticesAndFaces.vertices`), and an AnyTensor containing the faces
    information (`VerticesAndFaces.faces`).
    """

    vertices: Optional[AnyTensor]
    faces: Optional[AnyTensor]

    @classmethod
    def validate(
        cls: Type[T],
        value: Union[str, Any],
    ) -> T:
        return super().validate(value)

    def display(self) -> None:
        """
        Plot mesh consisting of vertices and faces.
        """
        import trimesh
        from IPython.display import display

        if self.vertices is None or self.faces is None:
            raise ValueError(
                'Can\'t display mesh from tensors when the vertices and/or faces '
                'are None.'
            )

        mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces)
        display(mesh.show())
