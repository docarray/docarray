from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    import trimesh

MESH_FILE_FORMATS = ('obj', 'glb', 'ply')


def load_trimesh_instance(
    uri: str, force: Optional[str] = None
) -> Union[trimesh.Trimesh, trimesh.Scene]:
    """
    Load the data from the url into a trimesh.Mesh or trimesh.Scene object.

    :param uri: uri to load data from
    :param force: str or None. For 'mesh' try to coerce scenes into a single mesh.
        For 'scene' try to coerce everything into a scene.
    :return: trimesh.Mesh or trimesh.Scene object
    """
    import urllib.parse

    scheme = urllib.parse.urlparse(uri).scheme
    loader = trimesh.load_remote if scheme in ['http', 'https'] else trimesh.load

    mesh = loader(uri, force=force)

    return mesh
