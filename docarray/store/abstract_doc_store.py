from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional, Type

from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from docarray import BaseDoc, DocList


class AbstractDocStore(ABC):
    @staticmethod
    @abstractmethod
    def list(namespace: str, show_table: bool) -> List[str]:
        """List all DocArrays in the specified backend at the namespace.

        :param namespace: The namespace to list
        :param show_table: If true, a table is printed to the console
        :return: A list of DocArray names
        """
        ...

    @staticmethod
    @abstractmethod
    def delete(name: str, missing_ok: bool) -> bool:
        """Delete the DocArray object at the specified name

        :param name: The name of the DocArray to delete
        :param missing_ok: If true, no error will be raised if the DocArray does not exist.
        :return: True if the DocArray was deleted, False if it did not exist.
        """
        ...

    @staticmethod
    @abstractmethod
    def push(
        da: 'DocList',
        name: str,
        public: bool,
        show_progress: bool,
        branding: Optional[Dict],
    ) -> Dict:
        """Push this DocArray to the specified name.

        :param da: The DocArray to push
        :param name: The name to push to
        :param public: Whether the DocArray should be publicly accessible
        :param show_progress: If true, a progress bar will be displayed.
        :param branding: Branding information to be stored with the DocArray
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
        :param public: Whether the DocArray should be publicly accessible
        :param show_progress: If true, a progress bar will be displayed.
        :param branding: Branding information to be stored with the DocArray
        """
        ...

    @staticmethod
    @abstractmethod
    def pull(
        da_cls: Type['DocList'],
        name: str,
        show_progress: bool,
        local_cache: bool,
    ) -> 'DocList':
        """Pull a DocArray from the specified name.

        :param da_cls: The DocArray class to instantiate
        :param name: The name to pull from
        :param show_progress: If true, a progress bar will be displayed.
        :param local_cache: If true, the DocArray will be cached locally
        :return: A DocArray
        """
        ...

    @staticmethod
    @abstractmethod
    def pull_stream(
        da_cls: Type['DocList'],
        name: str,
        show_progress: bool,
        local_cache: bool,
    ) -> Iterator['BaseDoc']:
        """Pull a stream of documents from the specified name.

        :param da_cls: The DocArray class to instantiate
        :param name: The name to pull from
        :param show_progress: If true, a progress bar will be displayed.
        :param local_cache: If true, the DocArray will be cached locally
        :return: An iterator of documents"""
        ...
