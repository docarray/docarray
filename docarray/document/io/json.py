import orjson

from docarray.typing.tensor.abstract_tensor import AbstractTensor


def _default_orjson(obj):
    """
    default option for orjson dumps.
    :param obj:
    :return: return a json compatible object
    """

    if isinstance(obj, AbstractTensor):
        return obj.__docarray_to_json_compatible__()
    else:
        return obj


def orjson_dumps(v, *, default=None):
    # orjson.dumps returns bytes, to match standard json.dumps we need to decode
    return orjson.dumps(
        v, default=_default_orjson, option=orjson.OPT_SERIALIZE_NUMPY
    ).decode()
