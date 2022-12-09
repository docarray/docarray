from typing import TypeVar

import numpy as np

from docarray.typing.url.any_url import AnyUrl
from docarray.typing.url.helper_3d_data import load_trimesh_instance

T = TypeVar('T', bound='PointCloudUrl')


class PointCloudUrl(AnyUrl):
    """
    URL to a .obj, .glb, or .ply file.
    Can be remote (web) URL, or a local file path.
    """

    def load(self: 'T', samples: int, multiple_geometries: bool = False) -> np.array:
        """
        Load the data from the url into a numpy.ndarray containing point cloud
        information.

        EXAMPLE USAGE

        .. code-block:: python

            import numpy as np
            from docarray import Document

            from docarray.typing import PointCloudUrl


            class MyDoc(Document):
                mesh_url: PointCloudUrl


            doc = MyDoc(mesh_url="toydata/tetrahedron.obj")

            point_cloud = doc.mesh_url.load(samples=100)

            assert isinstance(point_cloud, np.ndarray)
            assert point_cloud.shape == (100, 3)

        :param samples: number of points to sample from the mesh
        :param multiple_geometries: when multiple geometry stored in one mesh file,
            then store geometries in a list.

        :return: itself after processed
        """
        point_cloud: np.ndarray

        if multiple_geometries:
            # try to coerce everything into a scene
            scene = load_trimesh_instance(uri=self, force='scene')
            point_cloud = np.stack(
                [np.array(geo.sample(samples)) for geo in scene.geometry.values()],
                axis=0,
            )
        else:
            # combine a scene into a single mesh
            mesh = load_trimesh_instance(uri=self, force='mesh')
            point_cloud = np.array(mesh.sample(samples))

        return point_cloud
