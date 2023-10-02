from typing import Any, Optional, Type, TypeVar, Union

from pydantic import Field

from docarray.base_doc import BaseDoc
from docarray.documents.mesh.vertices_and_faces import VerticesAndFaces
from docarray.typing.tensor.embedding import AnyEmbedding
from docarray.typing.url.url_3d.mesh_url import Mesh3DUrl
from docarray.utils._internal.pydantic import is_pydantic_v2

if is_pydantic_v2:
    from pydantic import model_validator

T = TypeVar('T', bound='Mesh3D')


class Mesh3D(BaseDoc):
    """
    Document for handling meshes for 3D data representation.

    A mesh is a representation for 3D data and contains vertices and faces information.
    Vertices are points in a 3D space, represented as a tensor of shape (n_points, 3).
    Faces are triangular surfaces that can be defined by three points in 3D space,
    corresponding to the three vertices of a triangle. Faces can be represented as a
    tensor of shape (n_faces, 3). Each number in that tensor refers to an index of a
    vertex in the tensor of vertices.

    The Mesh3D Document can contain:

    - an [`Mesh3DUrl`][docarray.typing.url.Mesh3DUrl] (`Mesh3D.url`)
    - a [`VerticesAndFaces`][docarray.documents.mesh.vertices_and_faces.VerticesAndFaces]
    object containing:

        - an [`AnyTensor`](../../../../api_references/typing/tensor/tensor) of
        vertices (`Mesh3D.tensors.vertices`)
        - an [`AnyTensor`](../../../../api_references/typing/tensor/tensor) of faces (`Mesh3D.tensors.faces`)

    - an [`AnyEmbedding`](../../../../api_references/typing/tensor/embedding) (`Mesh3D.embedding`)
    - a `bytes` object (`Mesh3D.bytes_`).

    You can use this Document directly:

    ```python
    from docarray.documents import Mesh3D

    # use it directly
    mesh = Mesh3D(url='https://people.sc.fsu.edu/~jburkardt/data/obj/al.obj')
    mesh.tensors = mesh.url.load()
    # model = MyEmbeddingModel()
    # mesh.embedding = model(mesh.tensors.vertices)
    ```

    You can extend this Document:

    ```python
    from docarray.documents import Mesh3D
    from docarray.typing import AnyEmbedding
    from typing import Optional


    # extend it
    class MyMesh3D(Mesh3D):
        name: Optional[str] = None


    mesh = MyMesh3D(url='https://people.sc.fsu.edu/~jburkardt/data/obj/al.obj')
    mesh.name = 'my first mesh'
    mesh.tensors = mesh.url.load()
    # model = MyEmbeddingModel()
    # mesh.embedding = model(mesh.vertices)
    ```

    You can use this Document for composition:

    ```python
    from docarray import BaseDoc
    from docarray.documents import Mesh3D, TextDoc


    # compose it
    class MultiModalDoc(BaseDoc):
        mesh: Mesh3D
        text: TextDoc


    mmdoc = MultiModalDoc(
        mesh=Mesh3D(url='https://people.sc.fsu.edu/~jburkardt/data/obj/al.obj'),
        text=TextDoc(text='hello world, how are you doing?'),
    )
    mmdoc.mesh.tensors = mmdoc.mesh.url.load()

    # or
    mmdoc.mesh.bytes_ = mmdoc.mesh.url.load_bytes()
    ```

    You can display your 3D mesh in a notebook from either its url, or its tensors:

    ```python
    from docarray.documents import Mesh3D

    # display from url
    mesh = Mesh3D(url='https://people.sc.fsu.edu/~jburkardt/data/obj/al.obj')
    # mesh.url.display()

    # display from tensors
    mesh.tensors = mesh.url.load()
    # mesh.tensors.display()
    ```

    """

    url: Optional[Mesh3DUrl] = Field(
        description='URL to a file containing 3D mesh information. Can be remote (web) URL, or a local file path.',
        example='https://people.sc.fsu.edu/~jburkardt/data/obj/al.obj',
        default=None,
    )
    tensors: Optional[VerticesAndFaces] = Field(
        description='A tensor object of 3D mesh of type `VerticesAndFaces`.',
        example=[[0, 1, 1], [1, 0, 1], [1, 1, 0]],
        default=None,
    )
    embedding: Optional[AnyEmbedding] = Field(
        description='Store an embedding: a vector representation of the 3D mesh.',
        default=[1, 0, 1],
    )
    bytes_: Optional[bytes] = Field(
        description='Bytes representation of 3D mesh.',
        default=None,
    )

    if is_pydantic_v2:

        @model_validator(mode='before')
        @classmethod
        def validate_model_before(cls, value):
            if isinstance(value, str):
                return {'url': value}
            return value

    else:

        @classmethod
        def validate(
            cls: Type[T],
            value: Union[str, Any],
        ) -> T:
            if isinstance(value, str):
                value = cls(url=value)
            return super().validate(value)
