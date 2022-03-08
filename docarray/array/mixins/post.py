from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ... import DocumentArray


class PostMixin:
    """Helper functions for posting DocumentArray to Jina Flow. """

    def post(self, host: str, show_progress: bool = False) -> 'DocumentArray':
        """Posting itself to a remote Flow/Sandbox and get the modified DocumentArray back

        :param host: a host string. Can be one of the following:
            - `grpc://192.168.0.123:8080/endpoint`
            - `websocket://192.168.0.123:8080/endpoint`
            - `http://192.168.0.123:8080/endpoint`
            - `jinahub://Hello/endpoint`
            - `jinahub+docker://Hello/endpoint`
            - `jinahub+sandbox://Hello/endpoint`

        :param show_progress: if to show a progressbar
        :return: the new DocumentArray returned from remote
        """

        from urllib.parse import urlparse

        r = urlparse(host)
        _on = r.path or '/'
        _port = r.port or None
        standardized_host = (
            r._replace(netloc=r.netloc.replace(f':{r.port}', ''))
            ._replace(path='')
            .geturl()
        )

        if r.scheme.startswith('jinahub'):
            from jina import Flow

            f = Flow(quiet=True).add(uses=standardized_host)
            with f:
                return f.post(_on, inputs=self, show_progress=show_progress)
        elif r.scheme in ('grpc', 'http', 'websocket'):
            if _port is None:
                raise ValueError(f'can not determine port from {host}')

            from jina import Client

            c = Client(host=r.hostname, port=_port, protocol=r.scheme)
            return c.post(_on, inputs=self, show_progress=show_progress)
        else:
            raise ValueError(f'unsupported scheme: {r.scheme}')
