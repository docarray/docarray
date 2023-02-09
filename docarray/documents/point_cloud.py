from typing import Any, Optional, Type, TypeVar, Union

import numpy as np

from docarray.base_document import BaseDocument
from docarray.typing import AnyEmbedding, AnyTensor, PointCloud3DUrl
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

    A PointCloud3D Document can contain an PointCloud3DUrl (`PointCloud3D.url`), an
    AnyTensor (`PointCloud3D.tensor`), and an AnyEmbedding (`PointCloud3D.embedding`).

    EXAMPLE USAGE:

    You can use this Document directly:

    .. code-block:: python

        from docarray.documents import PointCloud3D

        # use it directly
        pc = PointCloud3D(url='https://people.sc.fsu.edu/~jburkardt/data/obj/al.obj')
        pc.tensor = pc.url.load(samples=100)
        model = MyEmbeddingModel()
        pc.embedding = model(pc.tensor)

    You can extend this Document:

    .. code-block:: python

        from docarray.documents import PointCloud3D
        from docarray.typing import AnyEmbedding
        from typing import Optional

        # extend it
        class MyPointCloud3D(PointCloud3D):
            second_embedding: Optional[AnyEmbedding]


        pc = MyPointCloud3D(url='https://people.sc.fsu.edu/~jburkardt/data/obj/al.obj')
        pc.tensor = pc.url.load(samples=100)
        model = MyEmbeddingModel()
        pc.embedding = model(pc.tensor)
        pc.second_embedding = model(pc.tensor)


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
        mmdoc.point_cloud.tensor = mmdoc.point_cloud.url.load(samples=100)

        # or

        mmdoc.point_cloud.bytes = mmdoc.point_cloud.url.load_bytes()

    """

    url: Optional[PointCloud3DUrl]
    tensor: Optional[AnyTensor]
    color_tensor: Optional[AnyTensor]
    embedding: Optional[AnyEmbedding]
    bytes: Optional[bytes]

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
            value = cls(tensor=value)

        return super().validate(value)

    def display(self, display_from: str = 'url', samples: int = 10000) -> None:
        """
        Plot interactive point cloud from :attr:`.tensor`
        :param display_from: display point cloud from either url or tensor.
        :param samples: number of points to sample from the mesh, will be ignored if
            displayed from tensor.
        """
        import trimesh
        from hubble.utils.notebook import is_notebook
        from IPython.display import display

        if display_from not in ['tensor', 'url']:
            raise ValueError(f'Expected one of ["tensor", "url"], got "{display_from}"')

        if getattr(self, display_from) is None:
            raise ValueError(
                f'Can not to display point cloud from {display_from} when the '
                f'{display_from} is None.'
            )

        if display_from == 'url':
            tensor = self.url.load(samples=samples)
            colors = np.tile(
                np.array([0, 0, 0]), (tensor.get_comp_backend().shape(tensor)[0], 1)
            )
        else:
            tensor = self.tensor
            comp_be = self.tensor.get_comp_backend()
            colors = (
                self.color_tensor
                if self.color_tensor
                else np.tile(
                    np.array([0, 0, 0]),
                    (comp_be.shape(tensor)[0], 1),
                )
            )

        pc = trimesh.points.PointCloud(vertices=tensor, colors=colors)

        if is_notebook():
            s = trimesh.Scene(geometry=pc)
            display(s.show())
        else:
            display(pc.show())
