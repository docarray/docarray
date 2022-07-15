import re
from collections import namedtuple
from typing import TYPE_CHECKING, Dict, NamedTuple, Optional
from urllib.parse import urlparse

if TYPE_CHECKING:
    from ... import DocumentArray

# To match versions v1, v1.1, v1.1.1, latest, v1.1.1-gpu
_VERSION_PATTERN = '(v\d.\d.\d|v\d.\d|v\d|latest)((-(c|g)pu)?)'

_ParsedHost = namedtuple('ParsedHost', 'on host port version scheme')


class PostMixin:
    """Helper functions for posting DocumentArray to Jina Flow."""

    def _parse_host(self, host: str) -> NamedTuple:
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
            r._replace(netloc=r.netloc.replace(f':{r.port}', ''))
            ._replace(path='')
            .geturl()
        )
        port = r.port or None
        version = None
        scheme = r.scheme
        version_match = re.search(_VERSION_PATTERN, r.path)
        if version_match:
            version = version_match.group(0)
            on = r.path[version_match.end(0) :]
            host = host + '/' + version
        return _ParsedHost(on=on, host=host, port=port, version=version, scheme=scheme)

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

        parsed_host = self._parse_host(host)

        batch_size = batch_size or len(self)

        scheme = parsed_host.scheme
        host = parsed_host.host
        tls = False

        if scheme in ('grpcs', 'https', 'wss'):
            scheme = scheme[:-1]
            tls = True

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
            raise ValueError(f'unsupported scheme: {scheme}')
