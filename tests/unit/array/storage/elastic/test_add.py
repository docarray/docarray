from docarray import Document
from docarray.array.storage.base.helper import Offset2ID
from docarray.array.storage.elastic.seqlike import SequenceLikeMixin


class MockESMixin(SequenceLikeMixin):
    def __delitem__(self):
        pass

    def __getitem__(self):
        pass

    def __setitem__(self):
        pass


def test_add_ids_that_already_exist():
    es_seq_mixin = MockESMixin()
    initial_ids = ["r0", "r1", "r2", "r3"]
    new_docs = [
        Document(id="r0"),
        Document(id="r3"),
        Document(id="r4"),
        Document(id="r5"),
    ]
    expected_ids_after_extend = ["r0", "r1", "r2", "r3", "r4", "r5"]

    es_seq_mixin._offset2ids = Offset2ID(ids=initial_ids)
    es_seq_mixin._upload_batch = lambda *args, **kwargs: None
    es_seq_mixin._save_offset2ids = lambda *args, **kwargs: None
    es_seq_mixin.extend(new_docs)

    assert es_seq_mixin._offset2ids.ids == expected_ids_after_extend
