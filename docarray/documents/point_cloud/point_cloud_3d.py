from typing import TYPE_CHECKING, Any, Optional, Type, TypeVar, Union

import numpy as np
from pydantic import Field

from docarray.base_doc import BaseDoc
from docarray.documents.point_cloud.points_and_colors import PointsAndColors
from docarray.typing import AnyEmbedding, PointCloud3DUrl
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.utils._internal.misc import import_library
from docarray.utils._internal.pydantic import is_pydantic_v2

if is_pydantic_v2:
    from pydantic import model_validator

if TYPE_CHECKING:
    import tensorflow as tf  # type: ignore
    import torch
else:
    tf = import_library('tensorflow', raise_error=False)
    torch = import_library('torch', raise_error=False)


T = TypeVar('T', bound='PointCloud3D')


class PointCloud3D(BaseDoc):
    """
    Document for handling point clouds for 3D data representation.

    Point cloud is a representation of a 3D mesh. It is made by repeatedly and uniformly
    sampling points within the surface of the 3D body. Compared to the mesh
    representation, the point cloud is a fixed size ndarray of shape `(n_samples, 3)` and
    hence easier for deep learning algorithms to handle.

    A PointCloud3D Document can contain:

    - a [`PointCloud3DUrl`][docarray.typing.url.PointCloud3DUrl] (`PointCloud3D.url`)
    - a [`PointsAndColors`][docarray.documents.point_cloud.points_and_colors.PointsAndColors] object (`PointCloud3D.tensors`)
    - an [`AnyEmbedding`](../../../../api_references/typing/tensor/embedding) (`PointCloud3D.embedding`)
    - a `bytes` object (`PointCloud3D.bytes_`)

    You can use this Document directly:

    ```python
    from docarray.documents import PointCloud3D

    # use it directly
    pc = PointCloud3D(url='https://people.sc.fsu.edu/~jburkardt/data/obj/al.obj')
    pc.tensors = pc.url.load(samples=100)
    # model = MyEmbeddingModel()
    # pc.embedding = model(pc.tensors.points)
    ```

    You can extend this Document:

    ```python
    from docarray.documents import PointCloud3D
    from docarray.typing import AnyEmbedding
    from typing import Optional


    # extend it
    class MyPointCloud3D(PointCloud3D):
        second_embedding: Optional[AnyEmbedding] = None


    pc = MyPointCloud3D(url='https://people.sc.fsu.edu/~jburkardt/data/obj/al.obj')
    pc.tensors = pc.url.load(samples=100)
    # model = MyEmbeddingModel()
    # pc.embedding = model(pc.tensors.points)
    # pc.second_embedding = model(pc.tensors.colors)
    ```

    You can use this Document for composition:

    ```python
    from docarray import BaseDoc
    from docarray.documents import PointCloud3D, TextDoc


    # compose it
    class MultiModalDoc(BaseDoc):
        point_cloud: PointCloud3D
        text: TextDoc


    mmdoc = MultiModalDoc(
        point_cloud=PointCloud3D(
            url='https://people.sc.fsu.edu/~jburkardt/data/obj/al.obj'
        ),
        text=TextDoc(text='hello world, how are you doing?'),
    )
    mmdoc.point_cloud.tensors = mmdoc.point_cloud.url.load(samples=100)

    # or
    mmdoc.point_cloud.bytes_ = mmdoc.point_cloud.url.load_bytes()
    ```

    You can display your point cloud from either its url, or its tensors:

    ```python
    from docarray.documents import PointCloud3D

    # display from url
    pc = PointCloud3D(url='https://people.sc.fsu.edu/~jburkardt/data/obj/al.obj')
    # pc.url.display()

    # display from tensors
    pc.tensors = pc.url.load(samples=10000)
    # pc.tensors.display()
    ```
    """

    url: Optional[PointCloud3DUrl] = Field(
        description='URL to a file containing point cloud information. Can be remote (web) URL, or a local file path.',
        example='https://people.sc.fsu.edu/~jburkardt/data/obj/al.obj',
        default=None,
    )
    tensors: Optional[PointsAndColors] = Field(
        description='A tensor object of 3D point cloud of type `PointsAndColors`.',
        example=[[0, 0, 1], [1, 0, 1], [0, 1, 1]],
        default=None,
    )
    embedding: Optional[AnyEmbedding] = Field(
        description='Store an embedding: a vector representation of 3D point cloud.',
        example=[1, 1, 1],
        default=None,
    )
    bytes_: Optional[bytes] = Field(
        description='Bytes representation of 3D point cloud.',
        default=None,
    )

    @classmethod
    def _validate(self, value: Union[str, AbstractTensor, Any]) -> Any:
        if isinstance(value, str):
            value = {'url': value}
        elif isinstance(value, (AbstractTensor, np.ndarray)) or (
            torch is not None
            and isinstance(value, torch.Tensor)
            or (tf is not None and isinstance(value, tf.Tensor))
        ):
            value = {'tensors': PointsAndColors(points=value)}

        return value

    if is_pydantic_v2:

        @model_validator(mode='before')
        @classmethod
        def validate_model_before(cls, value):
            return cls._validate(value)

    else:

        @classmethod
        def validate(
            cls: Type[T],
            value: Union[str, AbstractTensor, Any],
        ) -> T:
            return super().validate(cls._validate(value))
