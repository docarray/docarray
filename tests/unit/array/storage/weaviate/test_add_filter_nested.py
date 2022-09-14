from email.quoprimime import quote
from docarray import Document, DocumentArray
from random import randrange
import numpy as np
import lorem

import pytest

@pytest.mark.filterwarnings('ignore::UserWarning')
def test_setting_of_name(start_storage):
    with pytest.raises(ValueError):
        da = DocumentArray(storage='weaviate')


@pytest.mark.filterwarnings('ignore::UserWarning')
def test_setting_of_name(start_storage):
    da = DocumentArray(storage='weaviate', config={'name': 'Document0'})

    filter = {'id': 'someRandomIdThatWillNotBeThere'}
    q = da.find(filter=filter)

    assert len(q) == 0


@pytest.mark.filterwarnings('ignore::UserWarning')
def test_adding_cross_refs(start_storage):
    da = DocumentArray(storage='weaviate', config={'name': 'Document1'})

    d1 = Document(text='Im the nested doc')
    d2 = Document(text='Im the 2nd nested doc')
    d3 = Document(text='Im the main doc', chunks=[d1, d2], id="d3")
    da.extend(
        [d1, d2, d3]
    )

    # Filter
    filter = {'id': 'd3'}
    q = da.find(filter=filter)

    assert q[0].text == 'Im the main doc'
    assert q[0].chunks[0].text == 'Im the nested doc'
    assert q[0].chunks[1].text == 'Im the 2nd nested doc'


@pytest.mark.filterwarnings('ignore::UserWarning')
def test_adding_large_amount_of_data_with_vectors(start_storage):
    da = DocumentArray(storage='weaviate', config={'name': 'Document2'})
    total_docs = 25_000
    lorum_arrays = []
    lorum_array = []
    i = 0
    while i < total_docs:
        lorum_array.append(Document(text=lorem.paragraph(), embedding=np.random.random([256]), granularity=randrange(9999)))
        if (i % 5_000) == 4_999:
            lorum_arrays.append(lorum_array)
            lorum_array = []
        i += 1
    for lorum_array in lorum_arrays:
        da.extend(
            lorum_array
        )

    assert len(da) == total_docs


@pytest.mark.filterwarnings('ignore::UserWarning')
def test_filter(start_storage):
    da = DocumentArray(storage='weaviate', config={'name': 'Document2'})

    filter = {'path': 'text', 'operator': 'Equal', 'valueText': 'aliquam'}
    q = da.find(filter=filter, limit=3)
    
    assert len(q) == 3
    assert ('aliquam' in q[0].text.lower()) == True


@pytest.mark.filterwarnings('ignore::UserWarning')
def test_vector_filter(start_storage):
    da = DocumentArray(storage='weaviate', config={'name': 'Document2'})

    q = Document(embedding=np.random.random([256]))
    q.match(da)
    
    assert len(q.matches) == 20
    

@pytest.mark.filterwarnings('ignore::UserWarning')
def test_vector_and_scalar_filter(start_storage):
    da = DocumentArray(storage='weaviate', config={'name': 'Document2'})
    
    filter = {'path': 'text', 'operator': 'Equal', 'valueText': 'aliquam'}
    embedding = np.random.random([256])
    q = da.find(embedding, filter=filter, limit=1)
    
    assert len(q) == 1
    assert ('aliquam' in q[0].text.lower()) == True


@pytest.mark.filterwarnings('ignore::UserWarning')
def test_vector_and_sorting_and_scalar_filter(start_storage):
    da = DocumentArray(storage='weaviate', config={'name': 'Document2'})

    filter = {'path': 'text', 'operator': 'Equal', 'valueText': 'aliquam'}
    embedding = np.random.random([256])
    sort = [{"path": ["granularity"], "order": "desc"}]  # or "asc"

    q = da.find(embedding, filter=filter, limit=10, sort=sort)
    
    assert len(q) == 10


@pytest.mark.filterwarnings('ignore::UserWarning')
def test_failing_when_mixing_vectors(start_storage):
    with pytest.raises(ValueError):
        da = DocumentArray(storage='weaviate', config={'name': 'Document3'})

        d1 = Document(text='Im the nested doc', embedding=[0.1, 0.2, 0.3])
        d2 = Document(text='Im the 2nd nested doc', embedding=[0.1, 0.2, 0.3, 0.4])
        d3 = Document(text='Im the main doc', chunks=[d1, d2], id="d3")

        da.extend(
            [d1, d2, d3]
        )
