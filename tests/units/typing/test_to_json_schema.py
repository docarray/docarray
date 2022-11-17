import pytest
from pydantic import schema_json_of

from docarray.typing import ID, AnyUrl, Embedding, ImageUrl, Tensor, TorchTensor


@pytest.mark.parametrize(
    'type_', [Tensor, Embedding, ImageUrl, AnyUrl, ID, TorchTensor]
)
def test_json(type_):
    # this test verify that all of our type can be dumped to json
    schema_json_of(type_)
