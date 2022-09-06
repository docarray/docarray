import pytest
import requests

from docarray import DocumentArray, Document


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
            'name': 'Test',
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
def test_cast_columns_qdrant(start_storage, type_da, type_column, request):

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
        },
    )

    docs = DocumentArray([Document(tags={'price': type_da(i)}) for i in range(10)])

    index.extend(docs)

    assert len(index) == N
