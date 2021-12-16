from typing import TYPE_CHECKING, Any
from .ndarray import flush_ndarray

if TYPE_CHECKING:
    from ..docarray_pb2 import DocumentProto
    from ... import Document


def flush_proto(pb_msg: 'DocumentProto', key: str, value: Any) -> None:
    try:
        if key == 'blob' or key == 'embedding':
            pb_msg = getattr(pb_msg, key)
            flush_ndarray(pb_msg, value)
        elif key == 'chunks' or key == 'matches':
            pb_msg.ClearField(key)
            for d in value:
                d: Document
                docs = getattr(pb_msg, key)
                docs.append(d.to_protobuf())
        elif key == 'tags':
            pb_msg.tags.Clear()
            pb_msg.tags.update(value)
        else:
            # other simple fields
            setattr(pb_msg, key, value)
    except Exception as ex:
        if len(ex.args) >= 1:
            ex.args = (f'Field `{key}`',) + ex.args
        raise
