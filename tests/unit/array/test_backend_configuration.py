import requests

from docarray import DocumentArray


def test_weaviate_hnsw(start_storage):
    da = DocumentArray(
        storage='weaviate',
        config={'n_dim': 100, 'ef': 100, 'ef_construction': 100, 'max_connections': 16},
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
