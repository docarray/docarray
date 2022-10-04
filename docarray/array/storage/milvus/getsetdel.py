from docarray.array.storage.base.getsetdel import BaseGetSetDelMixin
from docarray import Document


class GetSetDelMixin(BaseGetSetDelMixin):
    def _get_doc_by_id(self, _id: str) -> 'Document':
        # to be implemented
        ...

    def _del_doc_by_id(self, _id: str):
        # to be implemented
        ...

    def _set_doc_by_id(self, _id: str, value: 'Document'):
        # to be implemented
        ...

    def _load_offset2ids(self):
        # to be implemented
        ...

    def _save_offset2ids(self):
        # to be implemented
        ...
