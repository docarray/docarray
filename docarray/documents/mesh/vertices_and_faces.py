# Licensed to the LF AI & Data foundation under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import TYPE_CHECKING, Any, Type, TypeVar, Union

from docarray.base_doc import BaseDoc
from docarray.typing.tensor.tensor import AnyTensor
from docarray.utils._internal.misc import import_library

T = TypeVar('T', bound='VerticesAndFaces')


class VerticesAndFaces(BaseDoc):
    """
    Document for handling the tensor data of a [`Mesh3D`][docarray.documents.mesh.Mesh3D] object.

    A VerticesAndFaces Document can contain:

    - an [`AnyTensor`](../../../../api_references/typing/tensor/tensor)
    containing the vertices information (`VerticesAndFaces.vertices`)
    - an [`AnyTensor`](../../../../api_references/typing/tensor/tensor)
    containing the faces information (`VerticesAndFaces.faces`)
    """

    vertices: AnyTensor
    faces: AnyTensor

    @classmethod
    def _docarray_validate(
        cls: Type[T],
        value: Union[str, Any],
    ) -> T:
        return super().validate(value)

    def display(self) -> None:
        """
        Plot mesh consisting of vertices and faces.
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
