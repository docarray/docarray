import os
import urllib.parse
import urllib.request
from contextlib import nullcontext


def _uri_to_blob(uri: str, timeout=None) -> bytes:
    """Convert uri to blob
    Internally it reads uri into blob.
    :param uri: the uri of Document
    :param timeout: timeout for urlopen. Only relevant if uri is not local
    :return: blob bytes.
    """
    if urllib.parse.urlparse(uri).scheme in {'http', 'https', 'data'}:
        req = urllib.request.Request(uri, headers={'User-Agent': 'Mozilla/5.0'})
        urlopen_kwargs = {'timeout': timeout} if timeout is not None else {}
        with urllib.request.urlopen(req, **urlopen_kwargs) as fp:
            return fp.read()
    elif os.path.exists(uri):
        with open(uri, 'rb') as fp:
            return fp.read()
    else:
        raise FileNotFoundError(f'`{uri}` is not a URL or a valid local path')


def _get_file_context(file):
    if hasattr(file, 'write'):
        file_ctx = nullcontext(file)
    else:
        file_ctx = open(file, 'wb')  # type: ignore

    return file_ctx


def _is_uri(value: str) -> bool:
    scheme = urllib.parse.urlparse(value).scheme
    return (
        (scheme in {'http', 'https'})
        or (scheme in {'data'})
        or os.path.exists(value)
        or os.access(os.path.dirname(value), os.W_OK)
    )


def _is_datauri(value: str) -> bool:
    scheme = urllib.parse.urlparse(value).scheme
    return scheme in {'data'}
