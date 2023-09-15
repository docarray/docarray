import logging
from abc import abstractmethod
from typing import (
    TYPE_CHECKING,
    Dict,
    Iterable,
    Iterator,
    Tuple,
    Type,
    TypeVar,
    cast,
)

from typing_extensions import Literal
from typing_inspect import get_args

PUSH_PULL_PROTOCOL = Literal['s3', 'file']
SUPPORTED_PUSH_PULL_PROTOCOLS = get_args(PUSH_PULL_PROTOCOL)

if TYPE_CHECKING:  # pragma: no cover
    from docarray import BaseDoc, DocList
    from docarray.store.abstract_doc_store import AbstractDocStore


SelfPushPullMixin = TypeVar('SelfPushPullMixin', bound='PushPullMixin')


class PushPullMixin(Iterable['BaseDoc']):
    """Mixin class for push/pull functionality."""

    __backends__: Dict[str, Type['AbstractDocStore']] = {}
    doc_type: Type['BaseDoc']

    @abstractmethod
    def __len__(self) -> int:
        ...

    @staticmethod
    def resolve_url(url: str) -> Tuple[PUSH_PULL_PROTOCOL, str]:
        """Resolve the URL to the correct protocol and name.
        :param url: url to resolve
        """
        protocol, name = url.split('://', 2)
        if protocol in SUPPORTED_PUSH_PULL_PROTOCOLS:
            protocol = cast(PUSH_PULL_PROTOCOL, protocol)
            return protocol, name
        else:
            raise ValueError(f'Unsupported protocol {protocol}')

    @classmethod
    def get_pushpull_backend(
        cls: Type[SelfPushPullMixin], protocol: PUSH_PULL_PROTOCOL
    ) -> Type['AbstractDocStore']:
        """
        Get the backend for the given protocol.

        :param protocol: the protocol to use, e.g. 'file', 's3'
        :return: the backend class
        """
        if protocol in cls.__backends__:
            return cls.__backends__[protocol]

        if protocol == 'file':
            from docarray.store.file import FileDocStore

            cls.__backends__[protocol] = FileDocStore
            logging.debug('Loaded Local Filesystem backend')
        elif protocol == 's3':
            from docarray.store.s3 import S3DocStore

            cls.__backends__[protocol] = S3DocStore
            logging.debug('Loaded S3 backend')
        else:
            raise NotImplementedError(f'protocol {protocol} not supported')

        return cls.__backends__[protocol]

    def push(
        self,
        url: str,
        show_progress: bool = False,
        **kwargs,
    ) -> Dict:
        """Push this `DocList` object to the specified url.

        :param url: url specifying the protocol and save name of the `DocList`. Should be of the form ``protocol://namespace/name``. e.g. ``s3://bucket/path/to/namespace/name``, ``file:///path/to/folder/name``
        :param show_progress: If true, a progress bar will be displayed.
        """
        logging.info(f'Pushing {len(self)} docs to {url}')
        protocol, name = self.__class__.resolve_url(url)
        return self.__class__.get_pushpull_backend(protocol).push(
            self, name, show_progress  # type: ignore
        )

    @classmethod
    def push_stream(
        cls: Type[SelfPushPullMixin],
        docs: Iterator['BaseDoc'],
        url: str,
        show_progress: bool = False,
    ) -> Dict:
        """Push a stream of documents to the specified url.

        :param docs: a stream of documents
        :param url: url specifying the protocol and save name of the `DocList`. Should be of the form ``protocol://namespace/name``. e.g. ``s3://bucket/path/to/namespace/name``, ``file:///path/to/folder/name``
        :param show_progress: If true, a progress bar will be displayed.
        """
        logging.info(f'Pushing stream to {url}')
        protocol, name = cls.resolve_url(url)
        return cls.get_pushpull_backend(protocol).push_stream(docs, name, show_progress)

    @classmethod
    def pull(
        cls: Type[SelfPushPullMixin],
        url: str,
        show_progress: bool = False,
        local_cache: bool = True,
    ) -> 'DocList':
        """Pull a `DocList` from the specified url.

        :param url: url specifying the protocol and save name of the `DocList`. Should be of the form ``protocol://namespace/name``. e.g. ``s3://bucket/path/to/namespace/name``, ``file:///path/to/folder/name``
        :param show_progress: if true, display a progress bar.
        :param local_cache: store the downloaded `DocList` to local folder
        :return: a `DocList` object
        """
        from docarray.base_doc import AnyDoc

        if cls.doc_type == AnyDoc:
            raise TypeError(
                'There is no document schema defined. '
                'Please specify the `DocList`\'s Document type using `DocList[MyDoc]`.'
            )

        logging.info(f'Pulling {url}')
        protocol, name = cls.resolve_url(url)
        return cls.get_pushpull_backend(protocol).pull(
            cls, name, show_progress, local_cache  # type: ignore
        )

    @classmethod
    def pull_stream(
        cls: Type[SelfPushPullMixin],
        url: str,
        show_progress: bool = False,
        local_cache: bool = False,
    ) -> Iterator['BaseDoc']:
        """Pull a stream of Documents from the specified url.

        :param url: url specifying the protocol and save name of the `DocList`. Should be of the form ``protocol://namespace/name``. e.g. ``s3://bucket/path/to/namespace/name``, ``file:///path/to/folder/name``
        :param show_progress: if true, display a progress bar.
        :param local_cache: store the downloaded `DocList` to local folder
        :return: Iterator of Documents
        """
        from docarray.base_doc import AnyDoc

        if cls.doc_type == AnyDoc:
            raise TypeError(
                'There is no document schema defined. '
                'Please specify the `DocList`\'s Document type using `DocList[MyDoc]`.'
            )

        logging.info(f'Pulling Document stream from {url}')
        protocol, name = cls.resolve_url(url)
        return cls.get_pushpull_backend(protocol).pull_stream(
            cls, name, show_progress, local_cache  # type: ignore
        )
