from typing import Any, Optional, Type, TypeVar, Union

import numpy as np

from docarray.base_document import BaseDocument
from docarray.documents.point_cloud.points_and_colors import PointsAndColors
from docarray.typing import AnyEmbedding, PointCloud3DUrl
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.utils.misc import is_tf_available, is_torch_available

torch_available = is_torch_available()
if torch_available:
    import torch

tf_available = is_tf_available()
if tf_available:
    import tensorflow as tf  # type: ignore

T = TypeVar('T', bound='PointCloud3D')


class PointCloud3D(BaseDocument):
    """
    Document for handling point clouds for 3D data representation.

    Point cloud is a representation of a 3D mesh. It is made by repeatedly and uniformly
    sampling points within the surface of the 3D body. Compared to the mesh
    representation, the point cloud is a fixed size ndarray (shape=(n_samples, 3)) and
    hence easier for deep learning algorithms to handle.

    A PointCloud3D Document can contain an PointCloud3DUrl (`PointCloud3D.url`),
    a PointsAndColors object (`PointCloud3D.tensors`), and an AnyEmbedding
    (`PointCloud3D.embedding`).

    EXAMPLE USAGE:

    You can use this Document directly:

    .. code-block:: python

        from docarray.documents import PointCloud3D

        # use it directly
        pc = PointCloud3D(url='https://people.sc.fsu.edu/~jburkardt/data/obj/al.obj')
        pc.tensors = pc.url.load(samples=100)
        model = MyEmbeddingModel()
        pc.embedding = model(pc.tensors.points)

    You can extend this Document:

    .. code-block:: python

        from docarray.documents import PointCloud3D
        from docarray.typing import AnyEmbedding
        from typing import Optional


        # extend it
        class MyPointCloud3D(PointCloud3D):
            second_embedding: Optional[AnyEmbedding]


        pc = MyPointCloud3D(url='https://people.sc.fsu.edu/~jburkardt/data/obj/al.obj')
        pc.tensors = pc.url.load(samples=100)
        model = MyEmbeddingModel()
        pc.embedding = model(pc.tensors.points)
        pc.second_embedding = model(pc.tensors.colors)


    You can use this Document for composition:

    .. code-block:: python

        from docarray import BaseDocument
        from docarray.documents import PointCloud3D, Text


        # compose it
        class MultiModalDoc(BaseDocument):
            point_cloud: PointCloud3D
            text: Text


        mmdoc = MultiModalDoc(
            point_cloud=PointCloud3D(
                url='https://people.sc.fsu.edu/~jburkardt/data/obj/al.obj'
            ),
            text=Text(text='hello world, how are you doing?'),
        )
        mmdoc.point_cloud.tensors = mmdoc.point_cloud.url.load(samples=100)

        # or

        mmdoc.point_cloud.bytes_ = mmdoc.point_cloud.url.load_bytes()


    You can display your point cloud from either its url, or its tensors:

    .. code-block:: python

        from docarray.documents import PointCloud3D

        # display from url
        pc = PointCloud3D(url='https://people.sc.fsu.edu/~jburkardt/data/obj/al.obj')
        pc.url.display()

        # display from tensors
        pc.tensors = pc.url.load(samples=10000)
        model = MyEmbeddingModel()
        pc.embedding = model(pc.tensors.points)

    """

    url: Optional[PointCloud3DUrl]
    tensors: Optional[PointsAndColors]
    embedding: Optional[AnyEmbedding]
    bytes_: Optional[bytes]

    @classmethod
    def validate(
        cls: Type[T],
        value: Union[str, AbstractTensor, Any],
    ) -> T:
        if isinstance(value, str):
            value = cls(url=value)
        elif isinstance(value, (AbstractTensor, np.ndarray)) or (
            torch_available
            and isinstance(value, torch.Tensor)
            or (tf_available and isinstance(value, tf.Tensor))
        ):
            value = cls(tensors=PointsAndColors(points=value))

        return super().validate(value)
