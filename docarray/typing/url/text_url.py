from typing import Optional, TYPE_CHECKING, TypeVar, Type, Union, Any

from docarray.typing.proto_register import _register_proto
from docarray.typing.url.any_url import AnyUrl
from docarray.typing.url.filetypes import TEXT_FILE_FORMATS

if TYPE_CHECKING:
    from pydantic import BaseConfig
    from pydantic.fields import ModelField

T = TypeVar('T', bound='TextUrl')


@_register_proto(proto_type_name='text_url')
class TextUrl(AnyUrl):
    """
    URL to a text file.
    Can be remote (web) URL, or a local file path.
    """

    @classmethod
    def validate(
        cls: Type[T],
        value: Union[T, str, Any],
        field: 'ModelField',
        config: 'BaseConfig',
    ):
        import os
        from urllib.parse import urlparse

        url = super().validate(value, field, config)  # basic url validation
        path = urlparse(url).path
        ext = os.path.splitext(path)[1][1:].lower()

        # pass test if extension is valid or no extension
        has_valid_text_extension = ext in TEXT_FILE_FORMATS or ext == ''

        if not has_valid_text_extension:
            raise ValueError('Text URL must have a valid extension')
        return cls(str(url), scheme=None)

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
        _bytes = self.load_bytes(timeout=timeout)
        return _bytes.decode(charset)
