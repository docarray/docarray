from typing import TYPE_CHECKING, Any, Dict, TypeVar

import numpy as np
from pydantic import parse_obj_as

from docarray.typing.proto_register import _register_proto
from docarray.typing.tensor.ndarray import NdArray
from docarray.typing.url.url_3d.url_3d import Url3D

if TYPE_CHECKING:
    from docarray.documents.point_cloud.points_and_colors import PointsAndColors


T = TypeVar('T', bound='PointCloud3DUrl')


@_register_proto(proto_type_name='point_cloud_url')
class PointCloud3DUrl(Url3D):
    """
    URL to a .obj, .glb, or .ply file containing point cloud information.
    Can be remote (web) URL, or a local file path.
    """

    def load(
        self: T,
        samples: int,
        multiple_geometries: bool = False,
        trimesh_args: Dict[str, Any] = None,
    ) -> 'PointsAndColors':
        """
        Load the data from the url into an NdArray containing point cloud information.

        EXAMPLE USAGE

        .. code-block:: python

            import numpy as np
            from docarray import BaseDocument

            from docarray.typing import PointCloud3DUrl


            class MyDoc(BaseDocument):
                point_cloud_url: PointCloud3DUrl


            doc = MyDoc(point_cloud_url="toydata/tetrahedron.obj")

            point_cloud = doc.point_cloud_url.load(samples=100)

            assert isinstance(point_cloud, np.ndarray)
            assert point_cloud.shape == (100, 3)

        :param samples: number of points to sample from the mesh
        :param multiple_geometries: if False, store point cloud in 2D np.ndarray.
            If True, store point clouds from multiple geometries in 3D np.ndarray.
        :param trimesh_args: dictionary of additional arguments for `trimesh.load()`
            or `trimesh.load_remote()`.

        :return: np.ndarray representing the point cloud
        """
        from docarray.documents.point_cloud.points_and_colors import PointsAndColors

        if multiple_geometries:
            # try to coerce everything into a scene
            scene = self._load_trimesh_instance(force='scene', **trimesh_args)
            point_cloud = np.stack(
                [np.array(geo.sample(samples)) for geo in scene.geometry.values()],
                axis=0,
            )
        else:
            # combine a scene into a single mesh
            mesh = self._load_trimesh_instance(force='mesh', **trimesh_args)
            point_cloud = np.array(mesh.sample(samples))

        points = parse_obj_as(NdArray, point_cloud)
        return PointsAndColors(points=points, colors=None)

    def display(
        self,
        samples: int = 10000,
        trimesh_args: Dict[str, Any] = None,
    ) -> None:
        """
        Plot point cloud from url.
        To use this you need to install trimesh[easy]: `pip install 'trimesh[easy]'`.

        First, it loads the point cloud into a :class:`PointsAndColors` object, and then
        calls display on it. The following is therefore equivalent:

        .. code-block:: python

            import numpy as np
            from docarray import BaseDocument

            from docarray.documents import PointCloud3D

            pc = PointCloud3D("toydata/tetrahedron.obj")

            # option 1
            pc.url.display()

            # option 2 (equivalent)
            pc.url.load(samples=10000).display()

        :param samples: number of points to sample from the mesh.
        :param trimesh_args: dictionary of additional arguments for `trimesh.load()`
            or `trimesh.load_remote()`.
        """
        self.load(samples=samples, **trimesh_args).display()
