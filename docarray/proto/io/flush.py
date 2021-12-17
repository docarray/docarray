from typing import TYPE_CHECKING

from google.protobuf.struct_pb2 import Struct

from .ndarray import flush_ndarray, read_ndarray
from ..docarray_pb2 import NdArrayProto, DocumentProto

if TYPE_CHECKING:
    from ... import Document


def parse_proto(pb_msg: 'DocumentProto') -> 'Document':
    from ... import Document
    fields = {}
    for (field, value) in pb_msg.ListFields():
        f_name = field.name
        if f_name == 'chunks' or f_name == 'matches':
            fields[f_name] = [Document.from_protobuf(d) for d in value]
        elif isinstance(value, NdArrayProto):
            fields[f_name] = read_ndarray(value)
        elif isinstance(value, Struct):
            fields[f_name] = dict(value)
        elif f_name == 'location':
            fields[f_name] = list(value)
        elif f_name == 'scores' or f_name == 'evaluations':
            ...
        else:
            fields[f_name] = value
    return Document(**fields)


def flush_proto(doc: 'Document') -> 'DocumentProto':
    pb_msg = DocumentProto()
    for key in doc.non_empty_fields:
        try:
            value = getattr(doc, key)
            if key == 'blob' or key == 'embedding':
                flush_ndarray(getattr(pb_msg, key), value)
            elif key == 'chunks' or key == 'matches':
                for d in value:
                    d: Document
                    docs = getattr(pb_msg, key)
                    docs.append(d.to_protobuf())
            elif key == 'tags':
                pb_msg.tags.update(value)
            elif key in ('scores', 'evaluations'):
                for kk, vv in value.items():
                    for ff in vv.non_empty_fields:
                        setattr(getattr(pb_msg, key)[kk], ff, getattr(vv, ff))
            else:
                # other simple fields
                setattr(pb_msg, key, value)
        except RecursionError as ex:
            if len(ex.args) >= 1:
                ex.args = (f'Field `{key}` contains cyclic reference in memory. '
                           f'Could it be your Document is referring to itself?',)
            raise
        except Exception as ex:
            if len(ex.args) >= 1:
                ex.args = (f'Field `{key}`',) + ex.args
            raise
    return pb_msg
