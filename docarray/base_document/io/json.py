import orjson

from docarray.typing.tensor.abstract_tensor import AbstractTensor


def _default_orjson(obj):
    """
    default option for orjson dumps.
    :param obj:
    :return: return a json compatible object
    """

    if isinstance(obj, AbstractTensor) or hasattr(obj, '_docarray_to_json_compatible'):
        return obj._docarray_to_json_compatible()
    else:
        return obj


def orjson_dumps(v, *, default=None) -> bytes:
    # dumps to bytes using orjson
    return orjson.dumps(v, default=_default_orjson, option=orjson.OPT_SERIALIZE_NUMPY)


def orjson_dumps_and_decode(v, *, default=None) -> str:
    # dumps to bytes using orjson
    return orjson_dumps(v, default=default).decode()
