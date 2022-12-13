from abc import ABC
from typing import TYPE_CHECKING, Any, Optional, Type, TypeVar, Union

import numpy as np

from docarray.typing.url.any_url import AnyUrl

if TYPE_CHECKING:
    import trimesh
    from pydantic import BaseConfig
    from pydantic.fields import ModelField

MESH_FILE_FORMATS = ('obj', 'glb', 'ply')

T = TypeVar('T', bound='Url3D')


class Url3D(AnyUrl, ABC):
    """
    URL to a .obj, .glb, or .ply file.
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
        self: T, force: Optional[str] = None
    ) -> Union['trimesh.Trimesh', 'trimesh.Scene']:
        """
        Load the data from the url into a trimesh.Mesh or trimesh.Scene object.

        :param url: url to load data from
        :param force: str or None. For 'mesh' try to coerce scenes into a single mesh.
            For 'scene' try to coerce everything into a scene.
        :return: trimesh.Mesh or trimesh.Scene object
        """
        import urllib.parse

        import trimesh

        scheme = urllib.parse.urlparse(self).scheme
        loader = trimesh.load_remote if scheme in ['http', 'https'] else trimesh.load

        mesh = loader(self, force=force)

        return mesh
