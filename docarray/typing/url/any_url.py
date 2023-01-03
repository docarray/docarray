from typing import TYPE_CHECKING, Type, TypeVar

from pydantic import AnyUrl as BaseAnyUrl
from pydantic import errors, parse_obj_as

from docarray.typing.abstract_type import AbstractType

if TYPE_CHECKING:
    from pydantic.networks import Parts

    from docarray.proto import NodeProto

T = TypeVar('T', bound='AnyUrl')


class AnyUrl(BaseAnyUrl, AbstractType):
    host_required = (
        False  # turn off host requirement to allow passing of local paths as URL
    )

    def _to_node_protobuf(self) -> 'NodeProto':
        """Convert Document into a NodeProto protobuf message. This function should
        be called when the Document is nested into another Document that need to
        be converted into a protobuf

        :return: the nested item protobuf message
        """
        from docarray.proto import NodeProto

        return NodeProto(any_url=str(self))

    @classmethod
    def validate_parts(cls, parts: 'Parts', validate_port: bool = True) -> 'Parts':
        """
        A method used to validate parts of a URL.
        Our URLs should be able to function both in local and remote settings.
        Therefore, we allow missing `scheme`, making it possible to pass a file
        path without prefix.
        If `scheme` is missing, we assume it is a local file path.
        """
        scheme = parts['scheme']
        if scheme is None:
            # assume local file if not otherwise stated, unlike pydantic
            scheme = 'file'
            parts['scheme'] = scheme

        elif cls.allowed_schemes and scheme.lower() not in cls.allowed_schemes:
            raise errors.UrlSchemePermittedError(set(cls.allowed_schemes))

        if validate_port:
            cls._validate_port(parts['port'])

        user = parts['user']
        if cls.user_required and user is None:
            raise errors.UrlUserInfoError()

        return parts

    @classmethod
    def from_protobuf(cls: Type[T], pb_msg: 'str') -> T:
        """
        read url from a proto msg
        :param pb_msg:
        :return: url
        """
        return parse_obj_as(cls, pb_msg)
