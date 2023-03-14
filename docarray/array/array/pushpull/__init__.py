import logging
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, Iterator, List, Optional, Type

from typing_extensions import Protocol

__cache_path__ = Path.home() / '.cache' / 'docarray-v2'

if TYPE_CHECKING:  # pragma: no cover
    from docarray import BaseDocument, DocumentArray


class ConcurrentPushException(Exception):
    """Exception raised when a concurrent push is detected."""

    pass


class PushPullLike(Protocol):
    @staticmethod
    def list(namespace: str, show_table: bool) -> List[str]:
        ...

    @staticmethod
    def delete(name: str, missing_ok: bool) -> bool:
        ...

    @staticmethod
    def push(
        da: 'DocumentArray',
        url: str,
        public: bool,
        show_progress: bool,
        branding: Optional[Dict],
    ) -> Dict:
        ...

    @staticmethod
    def push_stream(
        docs: Iterator['BaseDocument'],
        url: str,
        public: bool = True,
        show_progress: bool = False,
        branding: Optional[Dict] = None,
    ) -> Dict:
        ...

    @staticmethod
    def pull(
        cls: Type['DocumentArray'],
        name: str,
        show_progress: bool,
        local_cache: bool,
    ) -> 'DocumentArray':
        ...

    @staticmethod
    def pull_stream(
        cls: Type['DocumentArray'],
        name: str,
        show_progress: bool,
        local_cache: bool,
    ) -> Iterator['BaseDocument']:
        ...


class PushPullMixin(Iterable['BaseDocument']):
    """Mixin class for push/pull functionality."""

    __backends__: Dict[str, PushPullLike] = {}
    document_type: Type['BaseDocument']

    @abstractmethod
    def __len__(self) -> int:
        ...

    @classmethod
    def get_backend(cls, protocol: str) -> PushPullLike:
        """
        Get the backend for the given protocol.

        :param protocol: the protocol to use, e.g. 'jinaai', 'file', 's3'
        :return: the backend class
        """
        if protocol in cls.__backends__:
            return cls.__backends__[protocol]

        if protocol == 'jinaai':
            from docarray.array.array.pushpull.jinaai import PushPullJAC

            cls.__backends__[protocol] = PushPullJAC
            logging.info('Loaded Jina AI Cloud backend')
        elif protocol == 'file':
            from docarray.array.array.pushpull.file import PushPullFile

            cls.__backends__[protocol] = PushPullFile
            logging.info('Loaded Local Filesystem backend')
        elif protocol == 's3':
            from docarray.array.array.pushpull.s3 import PushPullS3

            cls.__backends__[protocol] = PushPullS3
            logging.info('Loaded S3 backend')
        else:
            raise NotImplementedError(f'protocol {protocol} not supported')

        return cls.__backends__[protocol]

    @staticmethod
    def list(
        url: str = f'file://{__cache_path__}', show_table: bool = False
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
        protocol, namespace = url.split('://', 2)
        return PushPullMixin.get_backend(protocol).list(namespace, show_table)

    @classmethod
    def delete(cls, url: str, missing_ok: bool = False):
        """
        Delete the DocumentArray at the given url.

        :param url: should be of the form ``protocol://namespace/name``. e.g. ``s3://bucket/path/to/namespace/name``, ``file:///path/to/folder/name``
        :param missing_ok: whether to ignore if the artifact does not exist
        """
        logging.info(f'Deleting artifact {url}')
        protocol, name = url.split('://', 2)
        success = PushPullMixin.get_backend(protocol).delete(
            name, missing_ok=missing_ok
        )
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
        protocol, name = url.split('://', 2)
        return PushPullMixin.get_backend(protocol).push(
            self, name, public, show_progress, branding  # type: ignore
        )

    @classmethod
    def push_stream(
        cls,
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
        protocol, name = url.split('://', 2)
        return PushPullMixin.get_backend(protocol).push_stream(
            docs, name, public, show_progress, branding
        )

    @classmethod
    def pull(
        cls,
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
        protocol, name = url.split('://', 2)
        return PushPullMixin.get_backend(protocol).pull(
            cls, name, show_progress, local_cache  # type: ignore
        )

    @classmethod
    def pull_stream(
        cls,
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
        protocol, name = url.split('://', 2)
        return PushPullMixin.get_backend(protocol).pull_stream(
            cls, name, show_progress, local_cache  # type: ignore
        )
