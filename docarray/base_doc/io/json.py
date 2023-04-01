import orjson
from pydantic.json import ENCODERS_BY_TYPE


def _default_orjson(obj):
    """
    default option for orjson dumps.
    :param obj:
    :return: return a json compatible object
    """
    from docarray.base_doc import BaseNode

    if isinstance(obj, BaseNode):
        return obj._docarray_to_json_compatible()
    else:
        for cls_, encoder in ENCODERS_BY_TYPE.items():
            if isinstance(obj, cls_):
                return encoder(obj)
        return obj


def orjson_dumps(v, *, default=None) -> bytes:
    # dumps to bytes using orjson
    return orjson.dumps(v, default=_default_orjson, option=orjson.OPT_SERIALIZE_NUMPY)


def orjson_dumps_and_decode(v, *, default=None) -> str:
    # dumps to bytes using orjson
    return orjson_dumps(v, default=default).decode()
