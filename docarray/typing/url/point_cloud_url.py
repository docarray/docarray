from typing import TYPE_CHECKING, Any, Type, TypeVar, Union

import numpy as np

from docarray.typing.url.any_url import AnyUrl
from docarray.typing.url.helper_3d_data import MESH_FILE_FORMATS, _load_trimesh_instance

if TYPE_CHECKING:
    from pydantic import BaseConfig
    from pydantic.fields import ModelField

    from docarray.proto import NodeProto

T = TypeVar('T', bound='PointCloudUrl')


class PointCloudUrl(AnyUrl):
    """
    URL to a .obj, .glb, or .ply file.
    Can be remote (web) URL, or a local file path.
    """

    def _to_node_protobuf(self) -> 'NodeProto':
        """Convert Document into a NodeProto protobuf message. This function should
        be called when the Document is nested into another Document that needs to
        be converted into a protobuf

        :return: the nested item protobuf message
        """
        from docarray.proto import NodeProto

        return NodeProto(point_cloud_url=str(self))

    @classmethod
    def validate(
        cls: Type[T],
        value: Union[T, np.ndarray, Any],
        field: 'ModelField',
        config: 'BaseConfig',
    ) -> T:
        url = super().validate(value, field, config)
        has_3d_extension = any(url.endswith(ext) for ext in MESH_FILE_FORMATS)
        if not has_3d_extension:
            raise ValueError(
                f'Point Cloud URL must have one of the following extensions:'
                f'{MESH_FILE_FORMATS}'
            )
        return cls(str(url), scheme=None)

    def load(self: 'T', samples: int, multiple_geometries: bool = False) -> np.ndarray:
        """
        Load the data from the url into a numpy.ndarray containing point cloud
        information.

        EXAMPLE USAGE

        .. code-block:: python

            import numpy as np
            from docarray import Document

            from docarray.typing import PointCloudUrl


            class MyDoc(Document):
                point_cloud_url: PointCloudUrl


            doc = MyDoc(point_cloud_url="toydata/tetrahedron.obj")

            point_cloud = doc.point_cloud_url.load(samples=100)

            assert isinstance(point_cloud, np.ndarray)
            assert point_cloud.shape == (100, 3)

        :param samples: number of points to sample from the mesh
        :param multiple_geometries: if False, store point cloud in 2D np.ndarray.
            If True, store point clouds from multiple geometries in 3D np.ndarray.

        :return: np.ndarray representing the point cloud
        """

        if multiple_geometries:
            # try to coerce everything into a scene
            scene = _load_trimesh_instance(url=self, force='scene')
            point_cloud = np.stack(
                [np.array(geo.sample(samples)) for geo in scene.geometry.values()],
                axis=0,
            )
        else:
            # combine a scene into a single mesh
            mesh = _load_trimesh_instance(url=self, force='mesh')
            point_cloud = np.array(mesh.sample(samples))

        return point_cloud
