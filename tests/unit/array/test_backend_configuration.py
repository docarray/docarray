import os
import random

import pytest
import requests

from docarray import Document, DocumentArray, dataclass
from docarray.typing import Image, Text


@dataclass
class MyDocument:
    image: Image
    paragraph: Text


cur_dir = os.path.dirname(os.path.abspath(__file__))


def test_weaviate_hnsw(start_storage):
    da = DocumentArray(
        storage='weaviate',
        config={
            'n_dim': 100,
            'ef': 100,
            'ef_construction': 100,
            'max_connections': 16,
            'dynamic_ef_min': 50,
            'dynamic_ef_max': 300,
            'dynamic_ef_factor': 4,
            'vector_cache_max_objects': 1000000,
            'flat_search_cutoff': 20000,
            'cleanup_interval_seconds': 1000,
            'skip': True,
            'distance': 'l2-squared',
        },
    )

    result = requests.get('http://localhost:8080/v1/schema').json()

    classes = result.get('classes', [])
    main_class = list(
        filter(lambda class_element: class_element['class'] == da._config.name, classes)
    )
    assert len(main_class) == 1

    main_class = main_class[0]

    assert main_class.get('vectorIndexConfig', {}).get('maxConnections') == 16
    assert main_class.get('vectorIndexConfig', {}).get('efConstruction') == 100
    assert main_class.get('vectorIndexConfig', {}).get('ef') == 100
    assert main_class.get('vectorIndexConfig', {}).get('dynamicEfMin') == 50
    assert main_class.get('vectorIndexConfig', {}).get('dynamicEfMax') == 300
    assert main_class.get('vectorIndexConfig', {}).get('dynamicEfFactor') == 4
    assert (
        main_class.get('vectorIndexConfig', {}).get('vectorCacheMaxObjects') == 1000000
    )
    assert main_class.get('vectorIndexConfig', {}).get('flatSearchCutoff') == 20000
    assert main_class.get('vectorIndexConfig', {}).get('cleanupIntervalSeconds') == 1000
    assert main_class.get('vectorIndexConfig', {}).get('skip') is True
    assert main_class.get('vectorIndexConfig', {}).get('distance') == 'l2-squared'


@pytest.mark.parametrize('columns', [[('price', 'int')], {'price': 'int'}])
def test_weaviate_da_w_protobuff(start_storage, columns):

    N = 10

    index = DocumentArray(
        storage='weaviate',
        config={
            'columns': columns,
        },
    )

    docs = DocumentArray([Document(tags={'price': i}) for i in range(N)])
    docs = DocumentArray.from_protobuf(
        docs.to_protobuf()
    )  # same as streaming the da in jina

    index.extend(docs)

    assert len(index) == N


@pytest.mark.parametrize('type_da', [int, float, str])
@pytest.mark.parametrize('type_column', ['int', 'float', 'str'])
def test_cast_columns_weaviate(start_storage, type_da, type_column, request):

    test_id = request.node.callspec.id.replace(
        '-', ''
    )  # remove '-' from the test id for the weaviate name
    N = 10

    index = DocumentArray(
        storage='weaviate',
        config={
            'name': f'Test{test_id}',
            'columns': {'price': type_column},
        },
    )

    docs = DocumentArray([Document(tags={'price': type_da(i)}) for i in range(10)])

    index.extend(docs)

    assert len(index) == N


@pytest.mark.parametrize('type_da', [int, float, str])
@pytest.mark.parametrize('type_column', ['int', 'float', 'str'])
def test_cast_columns_annlite(start_storage, type_da, type_column):

    N = 10

    index = DocumentArray(
        storage='annlite',
        config={
            'n_dim': 3,
            'columns': {'price': type_column},
        },
    )

    docs = DocumentArray([Document(tags={'price': type_da(i)}) for i in range(10)])

    index.extend(docs)

    assert len(index) == N


@pytest.mark.parametrize('type_da', [int, float, str])
@pytest.mark.parametrize('type_column', ['int', 'float', 'str'])
@pytest.mark.parametrize('prefer_grpc', [False, True])
def test_cast_columns_qdrant(start_storage, type_da, type_column, prefer_grpc, request):

    test_id = request.node.callspec.id.replace(
        '-', ''
    )  # remove '-' from the test id for the weaviate name
    N = 10

    index = DocumentArray(
        storage='qdrant',
        config={
            'collection_name': f'test{test_id}',
            'n_dim': 3,
            'columns': {'price': type_column},
            'prefer_grpc': prefer_grpc,
        },
    )

    docs = DocumentArray([Document(tags={'price': type_da(i)}) for i in range(10)])

    index.extend(docs)

    assert len(index) == N


@pytest.mark.parametrize('type_da', [int, float, str, bool])
@pytest.mark.parametrize('type_column', ['int', 'str', 'float', 'double', 'bool'])
def test_cast_columns_milvus(start_storage, type_da, type_column, request):
    test_id = request.node.callspec.id.replace(
        '-', ''
    )  # remove '-' from the test id for the milvus name
    N = 10

    index = DocumentArray(
        storage='milvus',
        config={
            'collection_name': f'test{test_id}',
            'n_dim': 3,
            'columns': {'price': type_column},
        },
    )

    docs = DocumentArray([Document(tags={'price': type_da(i)}) for i in range(N)])

    index.extend(docs)

    assert len(index) == N


def test_random_subindices_config():
    database_index = random.randint(0, 100)
    database_name = "jina" + str(database_index) + ".db"
    table_index = random.randint(0, 100)
    table_name = "test" + str(table_index)
    subindice_image_index = random.randint(0, 100)
    subindice_image_name = "test" + str(subindice_image_index)
    subindice_paragraph_index = random.randint(0, 100)
    subindice_paragraph_name = "test" + str(subindice_paragraph_index)
    sqlite3_config = {'connection': database_name, 'table_name': table_name}

    common_subindex_config = {
        '@.[image]': {'connection': database_name, 'table_name': subindice_image_name},
        '@.[paragraph]': {
            'connection': database_name,
            'table_name': subindice_paragraph_name,
        },
    }
    # extend with Documents, including embeddings
    _docs = [
        (
            MyDocument(
                image=os.path.join(cur_dir, '../document/toydata/test.png'),
                paragraph='hello world',
            )
        )
    ]

    da = DocumentArray(
        storage='sqlite',  # use SQLite as vector database
        config=sqlite3_config,
        subindex_configs=common_subindex_config,  # set up subindices for image and description
    )
    da.summary()

    for item in _docs:
        d = Document(item)
        da.append(d)

    da = DocumentArray(
        storage='sqlite',  # use SQLite as vector database
        config=sqlite3_config,
        subindex_configs=common_subindex_config,  # set up subindices for image and description
    )
    da.summary()
