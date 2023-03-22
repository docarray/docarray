import logging
from abc import abstractmethod
from typing import (
    TYPE_CHECKING,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    cast,
)

from typing_extensions import Literal
from typing_inspect import get_args

from docarray.utils.cache import get_cache_path

PUSH_PULL_PROTOCOL = Literal['jinaai', 's3', 'file']
SUPPORTED_PUSH_PULL_PROTOCOLS = get_args(PUSH_PULL_PROTOCOL)

if TYPE_CHECKING:  # pragma: no cover
    from docarray import BaseDocument, DocumentArray
    from docarray.array.array.pushpull.abstract_doc_store import AbstractDocStore


class ConcurrentPushException(Exception):
    """Exception raised when a concurrent push is detected."""

    pass


SelfPushPullMixin = TypeVar('SelfPushPullMixin', bound='PushPullMixin')


class PushPullMixin(Iterable['BaseDocument']):
    """Mixin class for push/pull functionality."""

    __backends__: Dict[str, Type['AbstractDocStore']] = {}
    document_type: Type['BaseDocument']

    @abstractmethod
    def __len__(self) -> int:
        ...

    @staticmethod
    def resolve_url(url: str) -> Tuple[PUSH_PULL_PROTOCOL, str]:
        """Resolve the URL to the correct protocol and name."""
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

        :param protocol: the protocol to use, e.g. 'jinaai', 'file', 's3'
        :return: the backend class
        """
        if protocol in cls.__backends__:
            return cls.__backends__[protocol]

        if protocol == 'jinaai':
            from docarray.array.array.pushpull.jinaai import JACDocStore

            cls.__backends__[protocol] = JACDocStore
            logging.debug('Loaded Jina AI Cloud backend')
        elif protocol == 'file':
            from docarray.array.array.pushpull.file import FileDocStore

            cls.__backends__[protocol] = FileDocStore
            logging.debug('Loaded Local Filesystem backend')
        elif protocol == 's3':
            from docarray.array.array.pushpull.s3 import S3DocStore

            cls.__backends__[protocol] = S3DocStore
            logging.debug('Loaded S3 backend')
        else:
            raise NotImplementedError(f'protocol {protocol} not supported')

        return cls.__backends__[protocol]

    @classmethod
    def list(
        cls: Type[SelfPushPullMixin],
        url: str = f'file://{get_cache_path()}',
        show_table: bool = False,
    ) -> List[str]:
        """
        List all the DocumentArrays in the namespace.
        url should be of the form ``protocol://namespace``

        If no url is provided, the DocumentArrays in the local cache will be listed.

        :param url: should be of the form ``protocol://namespace``. e.g. ``s3://bucket/path/to/namespace``, ``file:///path/to/folder``
        :param show_table: whether to show the table of artifacts
        :return: a list of artifact names
        """
        logging.info(f'Listing artifacts from {url}')
        protocol, namespace = cls.resolve_url(url)
        return cls.get_pushpull_backend(protocol).list(namespace, show_table)

    @classmethod
    def delete(
        cls: Type[SelfPushPullMixin], url: str, missing_ok: bool = False
    ) -> bool:
        """
        Delete the DocumentArray at the given url.

        :param url: should be of the form ``protocol://namespace/name``. e.g. ``s3://bucket/path/to/namespace/name``, ``file:///path/to/folder/name``
        :param missing_ok: whether to ignore if the artifact does not exist
        :return: whether the artifact was successfully deleted
        """
        logging.info(f'Deleting artifact {url}')
        protocol, name = cls.resolve_url(url)
        success = cls.get_pushpull_backend(protocol).delete(name, missing_ok=missing_ok)
        if success:
            logging.info(f'Successfully deleted artifact {url}')
        else:
            logging.warning(f'Failed to delete artifact {url}')
        return success

    def push(
        self,
        url: str,
        public: bool = True,
        show_progress: bool = False,
        branding: Optional[Dict] = None,
    ) -> Dict:
        """Push this DocumentArray object to the specified url.

        :param url: url specifying the protocol and save name of the DocumentArray. Should be of the form ``protocol://namespace/name``. e.g. ``s3://bucket/path/to/namespace/name``, ``file:///path/to/folder/name``
        :param public:  Only used by ``jinaai`` protocol. If true, anyone can pull a DocumentArray if they know its name.
            Setting this to false will restrict access to only the creator.
        :param show_progress: If true, a progress bar will be displayed.
        :param branding: Only used by ``jinaai`` protocol. A dictionary of branding information to be sent to Jina AI Cloud. {"icon": "emoji", "background": "#fff"}
        """
        logging.info(f'Pushing {len(self)} docs to {url}')
        protocol, name = self.__class__.resolve_url(url)
        return self.__class__.get_pushpull_backend(protocol).push(
            self, name, public, show_progress, branding  # type: ignore
        )

    @classmethod
    def push_stream(
        cls: Type[SelfPushPullMixin],
        docs: Iterator['BaseDocument'],
        url: str,
        public: bool = True,
        show_progress: bool = False,
        branding: Optional[Dict] = None,
    ) -> Dict:
        """Push a stream of documents to the specified url.

        :param docs: a stream of documents
        :param url: url specifying the protocol and save name of the DocumentArray. Should be of the form ``protocol://namespace/name``. e.g. ``s3://bucket/path/to/namespace/name``, ``file:///path/to/folder/name``
        :param public:  Only used by ``jinaai`` protocol. If true, anyone can pull a DocumentArray if they know its name.
        :param show_progress: If true, a progress bar will be displayed.
        :param branding: Only used by ``jinaai`` protocol. A dictionary of branding information to be sent to Jina AI Cloud. {"icon": "emoji", "background": "#fff"}
        """
        logging.info(f'Pushing stream to {url}')
        protocol, name = cls.resolve_url(url)
        return cls.get_pushpull_backend(protocol).push_stream(
            docs, name, public, show_progress, branding
        )

    @classmethod
    def pull(
        cls: Type[SelfPushPullMixin],
        url: str,
        show_progress: bool = False,
        local_cache: bool = True,
    ) -> 'DocumentArray':
        """Pull a :class:`DocumentArray` from the specified url.

        :param url: url specifying the protocol and save name of the DocumentArray. Should be of the form ``protocol://namespace/name``. e.g. ``s3://bucket/path/to/namespace/name``, ``file:///path/to/folder/name``
        :param show_progress: if true, display a progress bar.
        :param local_cache: store the downloaded DocumentArray to local folder
        :return: a :class:`DocumentArray` object
        """
        from docarray.base_document import AnyDocument

        if cls.document_type == AnyDocument:
            raise TypeError(
                'There is no document schema defined. '
                'Please specify the DocumentArray\'s Document type using `DocumentArray[MyDoc]`.'
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
    ) -> Iterator['BaseDocument']:
        """Pull a stream of Documents from the specified url.

        :param url: url specifying the protocol and save name of the DocumentArray. Should be of the form ``protocol://namespace/name``. e.g. ``s3://bucket/path/to/namespace/name``, ``file:///path/to/folder/name``
        :param show_progress: if true, display a progress bar.
        :param local_cache: store the downloaded DocumentArray to local folder
        :return: Iterator of Documents
        """
        from docarray.base_document import AnyDocument

        if cls.document_type == AnyDocument:
            raise TypeError(
                'There is no document schema defined. '
                'Please specify the DocumentArray\'s Document type using `DocumentArray[MyDoc]`.'
            )

        logging.info(f'Pulling Document stream from {url}')
        protocol, name = cls.resolve_url(url)
        return cls.get_pushpull_backend(protocol).pull_stream(
            cls, name, show_progress, local_cache  # type: ignore
        )
