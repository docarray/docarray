from collections import namedtuple
from typing import TYPE_CHECKING, Dict, NamedTuple, Optional
from urllib.parse import urlparse

if TYPE_CHECKING:  # pragma: no cover
    from docarray import DocumentArray


_ParsedHost = namedtuple('ParsedHost', 'on host port version scheme')


def _parse_host(host: str) -> NamedTuple:
    """Parse a host string into namedtuple object.

    A parsed host's components are `on`, `host`, `port`, `version`, `scheme`.
    :param host: a host string. Can be one of the following:
        - `grpc://192.168.0.123:8080/endpoint`
        - `ws://192.168.0.123:8080/endpoint`
        - `http://192.168.0.123:8080/endpoint`
        - `jinahub://Hello/endpoint`
        - `jinahub+docker://Hello/endpoint`
        - `jinahub+docker://Hello/v0.0.1/endpoint`
        - `jinahub+docker://Hello/latest/endpoint`
        - `jinahub+sandbox://Hello/endpoint`
    """
    r = urlparse(host)
    on = r.path or '/'
    host = (
        r._replace(netloc=r.netloc.replace(f':{r.port}', ''))._replace(path='').geturl()
    )
    port = r.port or None
    version = None
    scheme = r.scheme
    splited_path = list(filter(None, r.path.split('/')))
    if len(splited_path) == 2:
        # path includes version and endpoint
        version = splited_path[0]
        host = host + '/' + version
        on = '/' + splited_path[1]

    return _ParsedHost(on=on, host=host, port=port, version=version, scheme=scheme)


class PostMixin:
    """Helper functions for posting DocumentArray to Jina Flow."""

    def post(
        self,
        host: str,
        show_progress: bool = False,
        batch_size: Optional[int] = None,
        parameters: Optional[Dict] = None,
        **kwargs,
    ) -> 'DocumentArray':
        """Posting itself to a remote Flow/Sandbox and get the modified DocumentArray back

        :param host: a host string. Can be one of the following:
            - `grpc://192.168.0.123:8080/endpoint`
            - `ws://192.168.0.123:8080/endpoint`
            - `http://192.168.0.123:8080/endpoint`
            - `jinahub://Hello/endpoint`
            - `jinahub+docker://Hello/endpoint`
            - `jinahub+docker://Hello/v0.0.1/endpoint`
            - `jinahub+docker://Hello/latest/endpoint`
            - `jinahub+sandbox://Hello/endpoint`

        :param show_progress: if to show a progressbar
        :param batch_size: number of Document on each request
        :param parameters: parameters to send in the request
        :return: the new DocumentArray returned from remote
        """

        if not self:
            return

        parsed_host = _parse_host(host)

        batch_size = batch_size or len(self)

        scheme = parsed_host.scheme
        host = parsed_host.host

        if scheme in ('grpcs', 'https', 'wss'):
            scheme = scheme[:-1]

        if scheme == 'ws':
            scheme = 'websocket'  # temp fix for the core

        if scheme.startswith('jinahub'):
            from jina import Flow

            f = Flow(quiet=True, prefetch=1).add(uses=host, **kwargs)
            with f:
                return f.post(
                    parsed_host.on,
                    inputs=self,
                    show_progress=show_progress,
                    request_size=batch_size,
                    parameters=parameters,
                    **kwargs,
                )
        elif scheme in ('grpc', 'http', 'ws', 'websocket'):
            from jina import Client

            if parsed_host.port:
                host += f':{parsed_host.port}'

            c = Client(host=host)
            return c.post(
                parsed_host.on,
                inputs=self,
                show_progress=show_progress,
                request_size=batch_size,
                parameters=parameters,
                **kwargs,
            )
        else:
            raise ValueError(f'unsupported scheme: `{scheme}`')
