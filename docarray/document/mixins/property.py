import mimetypes
from collections import defaultdict
from typing import TYPE_CHECKING, Optional, Dict, Union

from ._property import _PropertyMixin

if TYPE_CHECKING:
    from ...types import DocumentContentType, ArrayType
    from ... import DocumentArray
    from ...array.chunk import ChunkArray
    from ...array.match import MatchArray
    from ...score import NamedScore


_all_mime_types = set(mimetypes.types_map.values())


class PropertyMixin(_PropertyMixin):
    def _clear_content(self):
        self._data.content = None
        self._data.text = None
        self._data.tensor = None
        self._data.blob = None

    @property
    def content(self) -> Optional['DocumentContentType']:
        ct = self.content_type
        if ct:
            return getattr(self, ct)

    @_PropertyMixin.text.setter
    def text(self, value: str):
        self._clear_content()
        self._data.text = value

    @_PropertyMixin.blob.setter
    def blob(self, value: bytes):
        self._clear_content()
        self._data.blob = value

    @_PropertyMixin.tensor.setter
    def tensor(self, value: 'ArrayType'):
        self._clear_content()
        self._data.tensor = value

    @content.setter
    def content(self, value: 'DocumentContentType'):
        self._clear_content()
        if isinstance(value, bytes):
            self._data.blob = value
        elif isinstance(value, str):
            self._data.text = value
        elif value is not None:
            self._data.tensor = value

    @_PropertyMixin.uri.setter
    def uri(self, value: str):
        if value:
            mime_type = mimetypes.guess_type(value)[0]

            if mime_type:
                self._data.mime_type = mime_type
        self._data.uri = value

    @_PropertyMixin.mime_type.setter
    def mime_type(self, value: str):
        if value and value not in _all_mime_types:
            # given but not recognizable, do best guess
            r = mimetypes.guess_type(f'*.{value}')[0]
            value = r or value

        self._data.mime_type = value

    @property
    def scores(self) -> Dict[str, Union['NamedScore', Dict]]:
        if self._data.scores is None:
            from ...score import NamedScore

            self._data.scores = defaultdict(NamedScore)
        return self._data.scores

    @property
    def evaluations(self) -> Dict[str, Union['NamedScore', Dict]]:
        if self._data.evaluations is None:
            from ...score import NamedScore

            self._data.evaluations = defaultdict(NamedScore)
        return self._data.evaluations

    @property
    def chunks(self) -> 'ChunkArray':
        if self._data.chunks is None:
            from ...array.chunk import ChunkArray

            self._data.chunks = ChunkArray(
                None, reference_doc=self._data._reference_doc
            )
        return self._data.chunks

    @chunks.setter
    def chunks(self, value: 'DocumentArray'):
        from ...array.chunk import ChunkArray

        if not isinstance(value, ChunkArray):
            value = ChunkArray(value, reference_doc=self._data._reference_doc)

        self._data.chunks = value

    @property
    def matches(self) -> 'MatchArray':
        if self._data.matches is None:
            from ...array.match import MatchArray

            self._data.matches = MatchArray(
                None, reference_doc=self._data._reference_doc
            )
        return self._data.matches

    @matches.setter
    def matches(self, value: 'DocumentArray'):
        from ...array.match import MatchArray

        if not isinstance(value, MatchArray):
            value = MatchArray(value, reference_doc=self._data._reference_doc)

        self._data.matches = value

    @property
    def content_type(self) -> Optional[str]:
        nf = self.non_empty_fields
        if 'text' in nf:
            return 'text'
        elif 'tensor' in nf:
            return 'tensor'
        elif 'blob' in nf:
            return 'blob'
