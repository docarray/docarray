from abc import ABC
from typing import TYPE_CHECKING, Any, Dict, Optional, Type, TypeVar, Union

import numpy as np

from docarray.typing.proto_register import _register_proto
from docarray.typing.url.any_url import AnyUrl

if TYPE_CHECKING:
    import trimesh
    from pydantic import BaseConfig
    from pydantic.fields import ModelField

MESH_FILE_FORMATS = ('obj', 'glb', 'ply')

T = TypeVar('T', bound='Url3D')


@_register_proto(proto_type_name='url3d')
class Url3D(AnyUrl, ABC):
    """
    URL to a .obj, .glb, or .ply file containing 3D mesh or point cloud information.
    Can be remote (web) URL, or a local file path.
    """

    @classmethod
    def validate(
        cls: Type[T],
        value: Union[T, np.ndarray, Any],
        field: 'ModelField',
        config: 'BaseConfig',
    ) -> T:
        url = super().validate(value, field, config)
        has_mesh_extension = any(url.endswith(ext) for ext in MESH_FILE_FORMATS)
        if not has_mesh_extension:
            raise ValueError(
                f'{cls.__name__} must have one of the following extensions:'
                f'{MESH_FILE_FORMATS}'
            )
        return cls(str(url), scheme=None)

    def _load_trimesh_instance(
        self: T,
        force: Optional[str] = None,
        trimesh_args: Optional[Dict[str, Any]] = None,
    ) -> Union['trimesh.Trimesh', 'trimesh.Scene']:
        """
        Load the data from the url into a trimesh.Mesh or trimesh.Scene object.

        :param force: str or None. For 'mesh' try to coerce scenes into a single mesh.
            For 'scene' try to coerce everything into a scene.
        :param trimesh_args: dictionary of additional arguments for `trimesh.load()`
            or `trimesh.load_remote()`.
        :return: trimesh.Mesh or trimesh.Scene object
        """
        import urllib.parse

        import trimesh

        if not trimesh_args:
            trimesh_args = {}

        scheme = urllib.parse.urlparse(self).scheme
        loader = trimesh.load_remote if scheme in ['http', 'https'] else trimesh.load

        mesh = loader(self, force=force, **trimesh_args)

        return mesh
