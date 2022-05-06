from typing import TYPE_CHECKING, Optional, Dict

if TYPE_CHECKING:
    from ... import DocumentArray


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
            - `jinahub+sandbox://Hello/endpoint`

        :param show_progress: if to show a progressbar
        :param batch_size: number of Document on each request
        :param parameters: parameters to send in the request
        :return: the new DocumentArray returned from remote
        """

        if not self:
            return

        from urllib.parse import urlparse

        r = urlparse(host)
        _on = r.path or '/'
        _port = r.port or None
        standardized_host = (
            r._replace(netloc=r.netloc.replace(f':{r.port}', ''))
            ._replace(path='')
            .geturl()
        )
        batch_size = batch_size or len(self)

        _scheme = r.scheme
        _tls = False

        if _scheme in ('grpcs', 'https', 'wss'):
            _scheme = _scheme[:-1]
            _tls = True

        if _scheme == 'ws':
            _scheme = 'websocket'  # temp fix for the core

        if _scheme.startswith('jinahub'):
            from jina import Flow

            f = Flow(quiet=True, prefetch=1).add(uses=standardized_host)
            with f:
                return f.post(
                    _on,
                    inputs=self,
                    show_progress=show_progress,
                    request_size=batch_size,
                    parameters=parameters,
                    **kwargs,
                )
        elif _scheme in ('grpc', 'http', 'ws', 'websocket'):
            from jina import Client

            if _port:
                standardized_host += f':{_port}'

            c = Client(host=standardized_host)
            return c.post(
                _on,
                inputs=self,
                show_progress=show_progress,
                request_size=batch_size,
                parameters=parameters,
                **kwargs,
            )
        else:
            raise ValueError(f'unsupported scheme: {r.scheme}')
