from typing import TYPE_CHECKING, Any, Optional, Type, TypeVar, Union

import numpy as np

from docarray.base_doc import BaseDoc
from docarray.typing import AnyTensor
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.utils._internal.misc import import_library

if TYPE_CHECKING:
    import tensorflow as tf  # type: ignore
    import torch
else:
    torch = import_library('torch', raise_error=False)
    tf = import_library('tensorflow', raise_error=False)


T = TypeVar('T', bound='PointsAndColors')


class PointsAndColors(BaseDoc):
    """
    Document for handling point clouds tensor data.

    A PointsAndColors Document can contain an AnyTensor containing the points in
    3D space information (`PointsAndColors.points`), and an AnyTensor containing
    the points' color information (`PointsAndColors.colors`).
    """

    points: AnyTensor
    colors: Optional[AnyTensor]

    @classmethod
    def validate(
        cls: Type[T],
        value: Union[str, AbstractTensor, Any],
    ) -> T:
        if isinstance(value, (AbstractTensor, np.ndarray)) or (
            torch is not None
            and isinstance(value, torch.Tensor)
            or (tf is not None and isinstance(value, tf.Tensor))
        ):
            value = cls(points=value)

        return super().validate(value)

    def display(self) -> None:
        """
        Plot point cloud consisting of points in 3D space and optionally colors.
        To use this you need to install trimesh[easy]: `pip install 'trimesh[easy]'`.
        """
        if TYPE_CHECKING:
            import trimesh
        else:
            trimesh = import_library('trimesh')
        from IPython.display import display

        colors = (
            self.colors
            if self.colors is not None
            else np.tile(
                np.array([0, 0, 0]),
                (self.points.get_comp_backend().shape(self.points)[0], 1),
            )
        )
        pc = trimesh.points.PointCloud(vertices=self.points, colors=colors)

        s = trimesh.Scene(geometry=pc)
        display(s.show())
