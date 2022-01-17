from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ...types import T


class MeshDataMixin:
    """Provide helper functions for :class:`Document` to support 3D mesh data and point cloud. """

    def load_uri_to_point_cloud_tensor(
        self: 'T', samples: int, as_chunks: bool = False
    ) -> 'T':
        """Convert a 3d mesh-like :attr:`.uri` into :attr:`.tensor`

        :param samples: number of points to sample from the mesh
        :param as_chunks: when multiple geometry stored in one mesh file,
            then store each geometry into different :attr:`.chunks`

        :return: itself after processed
        """
        import trimesh

        if as_chunks:
            from .. import Document

            # try to coerce everything into a scene
            scene = trimesh.load(self.uri, force='scene')
            for geo in scene.geometry.values():
                geo: trimesh.Trimesh
                self.chunks.append(Document(tensor=geo.sample(samples)))
        else:
            # combine a scene into a single mesh
            mesh = trimesh.load(self.uri, force='mesh')
            self.tensor = mesh.sample(samples)

        return self
