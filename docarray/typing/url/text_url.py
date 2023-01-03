from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from docarray.proto import NodeProto

from docarray.typing.url.any_url import AnyUrl
from docarray.typing.url.helper import _uri_to_blob


class TextUrl(AnyUrl):
    """
    URL to a text file.
    Can be remote (web) URL, or a local file path.
    """

    def _to_node_protobuf(self) -> 'NodeProto':
        """Convert Document into a NodeProto protobuf message. This function should
        be called when the Document is nested into another Document that need to
        be converted into a protobuf

        :return: the nested item protobuf message
        """
        from docarray.proto import NodeProto

        return NodeProto(text_url=str(self))

    def load_to_bytes(self, timeout: Optional[float] = None) -> bytes:
        """
        Load the text file into a bytes object.

        EXAMPLE USAGE

        .. code-block:: python

            from docarray import BaseDocument
            from docarray.typing import TextUrl


            class MyDoc(BaseDocument):
                remote_url: TextUrl
                local_url: TextUrl


            doc = MyDoc(
                remote_url='https://de.wikipedia.org/wiki/Brixen',
                local_url='home/username/my_file.txt',
            )

            remote_txt_bytes = doc.remote_url.load_to_bytes()
            local_txt_bytes = doc.local_url.load_to_bytes()

        :param timeout: timeout (sec) for urlopen network request.
            Only relevant if URL is not local
        :return: the text file content as bytes
        """
        return _uri_to_blob(self, timeout=timeout)

    def load(self, charset: str = 'utf-8', timeout: Optional[float] = None) -> str:
        """
        Load the text file into a string.

        EXAMPLE USAGE

        .. code-block:: python

            from docarray import BaseDocument
            from docarray.typing import TextUrl


            class MyDoc(BaseDocument):
                remote_url: TextUrl
                local_url: TextUrl


            doc = MyDoc(
                remote_url='https://de.wikipedia.org/wiki/Brixen',
                local_url='home/username/my_file.txt',
            )

            remote_txt = doc.remote_url.load()
            print(remote_txt)
            # prints: ```<!DOCTYPE html>\n<html class="client-nojs" ... > ...```

            local_txt = doc.local_url.load()
            print(local_txt)
            # prints content of my_file.txt


        :param timeout: timeout (sec) for urlopen network request.
            Only relevant if URL is not local
        :param charset: decoding charset; may be any character set registered with IANA
        :return: the text file content
        """
        _bytes = _uri_to_blob(self, timeout=timeout)
        return _bytes.decode(charset)
