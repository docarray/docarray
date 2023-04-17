from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional, Type

from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from docarray import BaseDoc, DocList


class AbstractDocStore(ABC):
    @staticmethod
    @abstractmethod
    def list(namespace: str, show_table: bool) -> List[str]:
        """List all DocLists in the specified backend at the namespace.

        :param namespace: The namespace to list
        :param show_table: If true, a table is printed to the console
        :return: A list of DocList names
        """
        ...

    @staticmethod
    @abstractmethod
    def delete(name: str, missing_ok: bool) -> bool:
        """Delete the DocList object at the specified name

        :param name: The name of the DocList to delete
        :param missing_ok: If true, no error will be raised if the DocList does not exist.
        :return: True if the DocList was deleted, False if it did not exist.
        """
        ...

    @staticmethod
    @abstractmethod
    def push(
        docs: 'DocList',
        name: str,
        public: bool,
        show_progress: bool,
        branding: Optional[Dict],
    ) -> Dict:
        """Push this DocList to the specified name.

        :param docs: The DocList to push
        :param name: The name to push to
        :param public: Whether the DocList should be publicly accessible
        :param show_progress: If true, a progress bar will be displayed.
        :param branding: Branding information to be stored with the DocList
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
        :param public: Whether the DocList should be publicly accessible
        :param show_progress: If true, a progress bar will be displayed.
        :param branding: Branding information to be stored with the DocList
        """
        ...

    @staticmethod
    @abstractmethod
    def pull(
        docs_cls: Type['DocList'],
        name: str,
        show_progress: bool,
        local_cache: bool,
    ) -> 'DocList':
        """Pull a DocList from the specified name.

        :param docs_cls: The DocList class to instantiate
        :param name: The name to pull from
        :param show_progress: If true, a progress bar will be displayed.
        :param local_cache: If true, the DocList will be cached locally
        :return: A DocList
        """
        ...

    @staticmethod
    @abstractmethod
    def pull_stream(
        docs_cls: Type['DocList'],
        name: str,
        show_progress: bool,
        local_cache: bool,
    ) -> Iterator['BaseDoc']:
        """Pull a stream of documents from the specified name.

        :param docs_cls: The DocList class to instantiate
        :param name: The name to pull from
        :param show_progress: If true, a progress bar will be displayed.
        :param local_cache: If true, the DocList will be cached locally
        :return: An iterator of documents"""
        ...
