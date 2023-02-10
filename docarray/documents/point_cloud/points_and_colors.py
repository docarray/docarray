from typing import Any, Optional, Type, TypeVar, Union

import numpy as np

from docarray.base_document import BaseDocument
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.typing.tensor.tensor import AnyTensor
from docarray.typing.url.url_3d.point_cloud_url import _display_point_cloud
from docarray.utils.misc import is_tf_available, is_torch_available

torch_available = is_torch_available()
if torch_available:
    import torch

tf_available = is_tf_available()
if tf_available:
    import tensorflow as tf  # type: ignore

T = TypeVar('T', bound='PointsAndColors')


class PointsAndColors(BaseDocument):
    """ """

    points: AnyTensor
    colors: Optional[AnyTensor]

    @classmethod
    def validate(
        cls: Type[T],
        value: Union[str, AbstractTensor, Any],
    ) -> T:
        if isinstance(value, (AbstractTensor, np.ndarray)) or (
            torch_available
            and isinstance(value, torch.Tensor)
            or (tf_available and isinstance(value, tf.Tensor))
        ):
            value = cls(points=value)

        return super().validate(value)

    def display(self) -> None:
        """
        Plot point cloud consisting of points in 3D space and optionally colors.
        """
        if self.points is None:
            raise ValueError(
                'Can\'t display point cloud from tensors when the points are None.'
            )
        _display_point_cloud(points=self.points, colors=self.colors)
