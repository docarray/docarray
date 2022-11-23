import orjson


def _default_orjson(obj):
    """
    default option for orjson dumps. It will call _to_json_compatible
    from docarray typing object that expose such method.
    :param obj:
    :return: return a json compatible object
    """

    if getattr(obj, '_to_json_compatible'):
        return obj._to_json_compatible()
    else:
        return obj


def orjson_dumps(v, *, default=None):
    # orjson.dumps returns bytes, to match standard json.dumps we need to decode
    return orjson.dumps(
        v, default=_default_orjson, option=orjson.OPT_SERIALIZE_NUMPY
    ).decode()
