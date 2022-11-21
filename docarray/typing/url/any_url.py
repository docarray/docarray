from typing import TYPE_CHECKING, Type, TypeVar

from pydantic import AnyUrl as BaseAnyUrl
from pydantic import errors, parse_obj_as

from docarray.document.base_node import BaseNode
from docarray.proto import NodeProto

if TYPE_CHECKING:
    from pydantic.networks import Parts

T = TypeVar('T', bound='AnyUrl')


class AnyUrl(BaseAnyUrl, BaseNode):
    host_required = (
        False  # turn off host requirement to allow passing of local paths as URL
    )

    def _to_node_protobuf(self) -> NodeProto:
        """Convert Document into a NodeProto protobuf message. This function should
        be called when the Document is nested into another Document that need to
        be converted into a protobuf

        :return: the nested item protobuf message
        """
        return NodeProto(any_url=str(self))

    @classmethod
    def validate_parts(cls, parts: 'Parts', validate_port: bool = True) -> 'Parts':
        """
        A method used to validate parts of a URL.
        Our URLs should be able to function both in local and remote settings.
        Therefore, we allow missing `scheme`, making it possible to pass a file path.
        """
        scheme = parts['scheme']
        if scheme is None:
            pass  # allow missing scheme, unlike pydantic

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
