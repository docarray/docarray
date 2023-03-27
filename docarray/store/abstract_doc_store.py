from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional, Type

from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from docarray import BaseDoc, DocumentArray


class AbstractDocStore(ABC):
    @staticmethod
    @abstractmethod
    def list(namespace: str, show_table: bool) -> List[str]:
        """List all DocumentArrays in the specified backend at the namespace.

        :param namespace: The namespace to list
        :param show_table: If true, a table is printed to the console
        :return: A list of DocumentArray names
        """
        ...

    @staticmethod
    @abstractmethod
    def delete(name: str, missing_ok: bool) -> bool:
        """Delete the DocumentArray object at the specified name

        :param name: The name of the DocumentArray to delete
        :param missing_ok: If true, no error will be raised if the DocumentArray does not exist.
        :return: True if the DocumentArray was deleted, False if it did not exist.
        """
        ...

    @staticmethod
    @abstractmethod
    def push(
        da: 'DocumentArray',
        name: str,
        public: bool,
        show_progress: bool,
        branding: Optional[Dict],
    ) -> Dict:
        """Push this DocumentArray to the specified name.

        :param da: The DocumentArray to push
        :param name: The name to push to
        :param public: Whether the DocumentArray should be publicly accessible
        :param show_progress: If true, a progress bar will be displayed.
        :param branding: Branding information to be stored with the DocumentArray
        """
        ...

    @staticmethod
    @abstractmethod
    def push_stream(
        docs: Iterator['BaseDoc'],
        url: str,
        public: bool = True,
        show_progress: bool = False,
        branding: Optional[Dict] = None,
    ) -> Dict:
        """Push a stream of documents to the specified name.

        :param docs: a stream of documents
        :param url: The name to push to
        :param public: Whether the DocumentArray should be publicly accessible
        :param show_progress: If true, a progress bar will be displayed.
        :param branding: Branding information to be stored with the DocumentArray
        """
        ...

    @staticmethod
    @abstractmethod
    def pull(
        da_cls: Type['DocumentArray'],
        name: str,
        show_progress: bool,
        local_cache: bool,
    ) -> 'DocumentArray':
        """Pull a DocumentArray from the specified name.

        :param da_cls: The DocumentArray class to instantiate
        :param name: The name to pull from
        :param show_progress: If true, a progress bar will be displayed.
        :param local_cache: If true, the DocumentArray will be cached locally
        :return: A DocumentArray
        """
        ...

    @staticmethod
    @abstractmethod
    def pull_stream(
        da_cls: Type['DocumentArray'],
        name: str,
        show_progress: bool,
        local_cache: bool,
    ) -> Iterator['BaseDoc']:
        """Pull a stream of documents from the specified name.

        :param da_cls: The DocumentArray class to instantiate
        :param name: The name to pull from
        :param show_progress: If true, a progress bar will be displayed.
        :param local_cache: If true, the DocumentArray will be cached locally
        :return: An iterator of documents"""
        ...
