from typing import TYPE_CHECKING, TypeVar

import numpy as np

from docarray.typing import NdArray
from docarray.typing.url.url_3d.url_3d import Url3D

if TYPE_CHECKING:
    from docarray.proto import NodeProto

T = TypeVar('T', bound='PointCloud3DUrl')


class PointCloud3DUrl(Url3D):
    """
    URL to a .obj, .glb, or .ply file containing point cloud information.
    Can be remote (web) URL, or a local file path.
    """

    def _to_node_protobuf(self: T) -> 'NodeProto':
        """Convert Document into a NodeProto protobuf message. This function should
        be called when the Document is nested into another Document that needs to
        be converted into a protobuf

        :return: the nested item protobuf message
        """
        from docarray.proto import NodeProto

        return NodeProto(point_cloud_url=str(self))

    def load(self: T, samples: int, multiple_geometries: bool = False) -> NdArray:
        """
        Load the data from the url into an NdArray containing point cloud information.

        EXAMPLE USAGE

        .. code-block:: python

            import numpy as np
            from docarray import BaseDocument

            from docarray.typing import PointCloud3DUrl


            class MyDoc(BaseDocument):
                point_cloud_url: PointCloud3DvUrl


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
            scene = self._load_trimesh_instance(force='scene')
            point_cloud = np.stack(
                [np.array(geo.sample(samples)) for geo in scene.geometry.values()],
                axis=0,
            )
        else:
            # combine a scene into a single mesh
            mesh = self._load_trimesh_instance(force='mesh')
            point_cloud = np.array(mesh.sample(samples))

        return point_cloud
