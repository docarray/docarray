from docarray.array.milvus import DocumentArrayMilvus, MilvusConfig
import pytest


@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArrayMilvus, lambda: MilvusConfig(n_dim=256)),
    ],
)
def test_from_files(da_cls, config, start_storage):
    assert (
        len(
            da_cls.from_files(patterns='*.*', to_dataturi=True, size=1, config=config())
        )
        == 1
    )
