import requests

from docarray import DocumentArray


def test_weaviate_hnsw(start_storage):
    da = DocumentArray(
        storage='weaviate',
        config={'n_dim': 100, 'ef': 100, 'ef_construction': 100, 'max_connections': 16,
                'dynamic_ef_min': 50, 'dynamic_ef_max': 300, 'dynamic_ef_factor': 4, 'vector_cache_max_objects': '1M',
                'flat_search_cutoff': 20000, 'cleanup_interval_seconds': 1000, 'skip': True},
    )

    result = requests.get('http://localhost:8080/v1/schema').json()

    classes = result.get('classes', [])
    assert len(classes) == 2
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
    assert main_class.get('vectorIndexConfig', {}).get('vectorCacheMaxObjects') == 1000000
    assert main_class.get('vectorIndexConfig', {}).get('flatSearchCutoff') == 20000
    assert main_class.get('vectorIndexConfig', {}).get('cleanupIntervalSeconds') == 1000
    assert main_class.get('vectorIndexConfig', {}).get('skip') is True
