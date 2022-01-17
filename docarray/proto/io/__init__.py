from collections import defaultdict
from typing import TYPE_CHECKING

from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct

from .ndarray import flush_ndarray, read_ndarray
from ..docarray_pb2 import NdArrayProto, DocumentProto

if TYPE_CHECKING:
    from ... import Document


def parse_proto(pb_msg: 'DocumentProto') -> 'Document':
    from ... import Document
    from ...score import NamedScore

    fields = {}
    for (field, value) in pb_msg.ListFields():
        f_name = field.name
        if f_name == 'chunks' or f_name == 'matches':
            fields[f_name] = [Document.from_protobuf(d) for d in value]
        elif isinstance(value, NdArrayProto):
            fields[f_name] = read_ndarray(value)
        elif isinstance(value, Struct):
            fields[f_name] = MessageToDict(value, preserving_proto_field_name=True)
        elif f_name == 'location':
            fields[f_name] = list(value)
        elif f_name == 'scores' or f_name == 'evaluations':
            fields[f_name] = defaultdict(NamedScore)
            for k, v in value.items():
                fields[f_name][k] = NamedScore(
                    {ff.name: vv for (ff, vv) in v.ListFields()}
                )
        else:
            fields[f_name] = value
    return Document(**fields)


def flush_proto(doc: 'Document') -> 'DocumentProto':
    pb_msg = DocumentProto()
    for key in doc.non_empty_fields:
        try:
            value = getattr(doc, key)
            if key in ('tensor', 'embedding'):
                flush_ndarray(getattr(pb_msg, key), value)
            elif key in ('chunks', 'matches'):
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
            elif key == 'location':
                pb_msg.location.extend(value)
            elif key == 'content':
                pass  # intentionally ignore `content` field as it is just a proxy
            else:
                # other simple fields
                setattr(pb_msg, key, value)
        except RecursionError as ex:
            if len(ex.args) >= 1:
                ex.args = (
                    f'Field `{key}` contains cyclic reference in memory. '
                    f'Could it be your Document is referring to itself?',
                )
            raise
        except Exception as ex:
            if len(ex.args) >= 1:
                ex.args = (f'Field `{key}` is problematic',) + ex.args
            raise
    return pb_msg
