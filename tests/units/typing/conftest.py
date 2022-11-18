from json import JSONEncoder

import pytest


@pytest.fixture(scope='module')
def json_encoder():
    class TestJsonEncoder(JSONEncoder):
        """
        This is a custom JSONEncoder that will call the
        _to_json_compatible method of type. This Encoder will be
        used when calling doc.json()
        """

        def default(self, obj):
            if hasattr(obj, '_to_json_compatible'):
                return obj._to_json_compatible()
            return JSONEncoder.default(self, obj)

    return TestJsonEncoder
