from abc import ABC
from typing import TYPE_CHECKING, Any, Dict, Optional, TypeVar, Union

from docarray.typing.proto_register import _register_proto
from docarray.typing.url.any_url import AnyUrl
from docarray.typing.url.mimetypes import OBJ_MIMETYPE
from docarray.utils._internal.misc import import_library

if TYPE_CHECKING:
    import trimesh

T = TypeVar('T', bound='Url3D')


@_register_proto(proto_type_name='url3d')
class Url3D(AnyUrl, ABC):
    """
    URL to a file containing 3D mesh or point cloud information.
    Can be remote (web) URL, or a local file path.
    """

    @classmethod
    def mime_type(cls) -> str:
        return OBJ_MIMETYPE

    def _load_trimesh_instance(
        self: T,
        force: Optional[str] = None,
        skip_materials: bool = True,
        trimesh_args: Optional[Dict[str, Any]] = None,
    ) -> Union['trimesh.Trimesh', 'trimesh.Scene']:
        """
        Load the data from the url into a trimesh.Mesh or trimesh.Scene object.

        :param force: str or None. For 'mesh' try to coerce scenes into a single mesh.
            For 'scene' try to coerce everything into a scene.
        :param skip_materials: Skip materials if True, else skip.
        :param trimesh_args: dictionary of additional arguments for `trimesh.load()`
            or `trimesh.load_remote()`.
        :return: trimesh.Mesh or trimesh.Scene object
        """
        import urllib.parse

        if TYPE_CHECKING:
            import trimesh
        else:
            trimesh = import_library('trimesh', raise_error=True)

        if not trimesh_args:
            trimesh_args = {}

        scheme = urllib.parse.urlparse(self).scheme
        loader = trimesh.load_remote if scheme in ['http', 'https'] else trimesh.load

        mesh = loader(self, force=force, skip_materials=skip_materials, **trimesh_args)

        return mesh
