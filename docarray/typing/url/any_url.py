import mimetypes
import os
import urllib
import urllib.parse
import urllib.request
from typing import TYPE_CHECKING, Any, List, Optional, Type, TypeVar, Union

import numpy as np
from pydantic import AnyUrl as BaseAnyUrl
from pydantic import errors, parse_obj_as

from docarray.typing.abstract_type import AbstractType
from docarray.typing.proto_register import _register_proto
from docarray.utils._internal.pydantic import is_pydantic_v2

if is_pydantic_v2:
    from pydantic_core import core_schema

if TYPE_CHECKING:
    if not is_pydantic_v2:
        from pydantic import BaseConfig
        from pydantic.fields import ModelField
    else:
        from pydantic import GetCoreSchemaHandler

    from pydantic.networks import Parts

    from docarray.proto import NodeProto

T = TypeVar('T', bound='AnyUrl')

mimetypes.init([])

# TODO need refactoring here
# - code is duplicate in both version
# - validation is very dummy for pydantic v2

if is_pydantic_v2:

    @_register_proto(proto_type_name='any_url')
    class AnyUrl(str, AbstractType):  # todo dummy url for now
        @classmethod
        def _docarray_validate(
            cls: Type[T],
            value: Any,
            _: Any,
        ):

            if not cls.is_extension_allowed(value):
                raise ValueError(
                    f"The file '{value}' is not in a valid format for class '{cls.__name__}'."
                )

            return cls(str(value))

        def __get_pydantic_core_schema__(
            cls, source: Type[Any], handler: Optional['GetCoreSchemaHandler'] = None
        ) -> core_schema.CoreSchema:
            return core_schema.general_after_validator_function(
                cls._docarray_validate,
                core_schema.str_schema(),
            )

        def load_bytes(self, timeout: Optional[float] = None) -> bytes:
            """Convert url to bytes. This will either load or download the file and save
            it into a bytes object.
            :param timeout: timeout for urlopen. Only relevant if URI is not local
            :return: bytes.
            """
            if urllib.parse.urlparse(self).scheme in {'http', 'https', 'data'}:
                req = urllib.request.Request(
                    self, headers={'User-Agent': 'Mozilla/5.0'}
                )
                urlopen_kwargs = {'timeout': timeout} if timeout is not None else {}
                with urllib.request.urlopen(req, **urlopen_kwargs) as fp:  # type: ignore
                    return fp.read()
            elif os.path.exists(self):
                with open(self, 'rb') as fp:
                    return fp.read()
            else:
                raise FileNotFoundError(f'`{self}` is not a URL or a valid local path')

        def _to_node_protobuf(self) -> 'NodeProto':
            """Convert Document into a NodeProto protobuf message. This function should
            be called when the Document is nested into another Document that need to
            be converted into a protobuf

            :return: the nested item protobuf message
            """
            from docarray.proto import NodeProto

            return NodeProto(text=str(self), type=self._proto_type_name)

        @classmethod
        def from_protobuf(cls: Type[T], pb_msg: 'str') -> T:
            """
            Read url from a proto msg.
            :param pb_msg:
            :return: url
            """
            return parse_obj_as(cls, pb_msg)

        @classmethod
        def is_extension_allowed(cls, value: Any) -> bool:
            """
            Check if the file extension of the URL is allowed for this class.
            First, it guesses the mime type of the file. If it fails to detect the
            mime type, it then checks the extra file extensions.
            Note: This method assumes that any URL without an extension is valid.

            :param value: The URL or file path.
            :return: True if the extension is allowed, False otherwise
            """
            if cls is AnyUrl:
                return True

            url_parts = value.split('?')
            extension = cls._get_url_extension(value)
            if not extension:
                return True

            mimetype, _ = mimetypes.guess_type(url_parts[0])
            if mimetype and mimetype.startswith(cls.mime_type()):
                return True

            return extension in cls.extra_extensions()

        @staticmethod
        def _get_url_extension(url: str) -> str:
            """
            Extracts and returns the file extension from a given URL.
            If no file extension is present, the function returns an empty string.


            :param url: The URL to extract the file extension from.
            :return: The file extension without the period, if one exists,
                otherwise an empty string.
            """

            parsed_url = urllib.parse.urlparse(url)
            ext = os.path.splitext(parsed_url.path)[1]
            ext = ext[1:] if ext.startswith('.') else ext
            return ext

else:

    @_register_proto(proto_type_name='any_url')
    class AnyUrl(BaseAnyUrl, AbstractType):
        host_required = (
            False  # turn off host requirement to allow passing of local paths as URL
        )

        @classmethod
        def mime_type(cls) -> str:
            """Returns the mime type associated with the class."""
            raise NotImplementedError

        @classmethod
        def extra_extensions(cls) -> List[str]:
            """Returns a list of allowed file extensions for the class
            that are not covered by the mimetypes library."""
            raise NotImplementedError

        def _to_node_protobuf(self) -> 'NodeProto':
            """Convert Document into a NodeProto protobuf message. This function should
            be called when the Document is nested into another Document that need to
            be converted into a protobuf

            :return: the nested item protobuf message
            """
            from docarray.proto import NodeProto

            return NodeProto(text=str(self), type=self._proto_type_name)

        @staticmethod
        def _get_url_extension(url: str) -> str:
            """
            Extracts and returns the file extension from a given URL.
            If no file extension is present, the function returns an empty string.


            :param url: The URL to extract the file extension from.
            :return: The file extension without the period, if one exists,
                otherwise an empty string.
            """

            parsed_url = urllib.parse.urlparse(url)
            ext = os.path.splitext(parsed_url.path)[1]
            ext = ext[1:] if ext.startswith('.') else ext
            return ext

        @classmethod
        def is_extension_allowed(cls, value: Any) -> bool:
            """
            Check if the file extension of the URL is allowed for this class.
            First, it guesses the mime type of the file. If it fails to detect the
            mime type, it then checks the extra file extensions.
            Note: This method assumes that any URL without an extension is valid.

            :param value: The URL or file path.
            :return: True if the extension is allowed, False otherwise
            """
            if cls is AnyUrl:
                return True

            url_parts = value.split('?')
            extension = cls._get_url_extension(value)
            if not extension:
                return True

            mimetype, _ = mimetypes.guess_type(url_parts[0])
            if mimetype and mimetype.startswith(cls.mime_type()):
                return True

            return extension in cls.extra_extensions()

        @classmethod
        def validate(
            cls: Type[T],
            value: Union[T, np.ndarray, Any],
            field: 'ModelField',
            config: 'BaseConfig',
        ) -> T:
            import os

            abs_path: Union[T, np.ndarray, Any]
            if (
                isinstance(value, str)
                and not value.startswith('http')
                and not os.path.isabs(value)
            ):
                input_is_relative_path = True
                abs_path = os.path.abspath(value)
            else:
                input_is_relative_path = False
                abs_path = value

            url = super().validate(abs_path, field, config)  # basic url validation

            if not cls.is_extension_allowed(value):
                raise ValueError(
                    f"The file '{value}' is not in a valid format for class '{cls.__name__}'."
                )

            return cls(str(value if input_is_relative_path else url), scheme=None)

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
                # allow missing scheme, unlike pydantic
                pass

            elif cls.allowed_schemes and scheme.lower() not in cls.allowed_schemes:
                raise errors.UrlSchemePermittedError(set(cls.allowed_schemes))

            if validate_port:
                cls._validate_port(parts['port'])

            user = parts['user']
            if cls.user_required and user is None:
                raise errors.UrlUserInfoError()

            return parts

        @classmethod
        def build(
            cls,
            *,
            scheme: str,
            user: Optional[str] = None,
            password: Optional[str] = None,
            host: str,
            port: Optional[str] = None,
            path: Optional[str] = None,
            query: Optional[str] = None,
            fragment: Optional[str] = None,
            **_kwargs: str,
        ) -> str:
            """
            Build a URL from its parts.
            The only difference from the pydantic implementation is that we allow
            missing `scheme`, making it possible to pass a file path without prefix.
            """

            # allow missing scheme, unlike pydantic
            scheme_ = scheme if scheme is not None else ''
            url = super().build(
                scheme=scheme_,
                user=user,
                password=password,
                host=host,
                port=port,
                path=path,
                query=query,
                fragment=fragment,
                **_kwargs,
            )
            if scheme is None and url.startswith('://'):
                # remove the `://` prefix, since scheme is missing
                url = url[3:]
            return url

        @classmethod
        def from_protobuf(cls: Type[T], pb_msg: 'str') -> T:
            """
            Read url from a proto msg.
            :param pb_msg:
            :return: url
            """
            return parse_obj_as(cls, pb_msg)

        def load_bytes(self, timeout: Optional[float] = None) -> bytes:
            """Convert url to bytes. This will either load or download the file and save
            it into a bytes object.
            :param timeout: timeout for urlopen. Only relevant if URI is not local
            :return: bytes.
            """
            if urllib.parse.urlparse(self).scheme in {'http', 'https', 'data'}:
                req = urllib.request.Request(
                    self, headers={'User-Agent': 'Mozilla/5.0'}
                )
                urlopen_kwargs = {'timeout': timeout} if timeout is not None else {}
                with urllib.request.urlopen(req, **urlopen_kwargs) as fp:  # type: ignore
                    return fp.read()
            elif os.path.exists(self):
                with open(self, 'rb') as fp:
                    return fp.read()
            else:
                raise FileNotFoundError(f'`{self}` is not a URL or a valid local path')
