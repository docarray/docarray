from docarray.array.queryset import QueryParser


def test_empty_query():
    query = QueryParser()
    assert query.lookup_groups is None


def test_simple_query():
    query = QueryParser({'tags__x': {'$gte': 2}})
    assert query.lookup_groups.lookups == {'tags__x__gte': 2}

    query = QueryParser({'tags__x': {'$gte': 2}, 'tags__y': {'$lt': 1}})
    assert len(query.lookup_groups.children) == 2
    assert query.lookup_groups.op == 'and'
    assert query.lookup_groups.children[1].lookups == {'tags__y__lt': 1}

    query = QueryParser({'tags__x': {'$gte': 0, '$lte': 54}})
    assert len(query.lookup_groups.children) == 2
    assert query.lookup_groups.op == 'and'
    assert query.lookup_groups.children[0].lookups == {'tags__x__gte': 0}
    assert query.lookup_groups.children[1].lookups == {'tags__x__lte': 54}


def test_logic_query():
    query = QueryParser({'$or': {'tags__x': {'$lt': 1}, 'tags__y': {'$gte': 50}}})
    assert len(query.lookup_groups.children) == 2
    assert query.lookup_groups.op == 'or'
    assert query.lookup_groups.children[0].lookups == {'tags__x__lt': 1}
    assert query.lookup_groups.children[1].lookups == {'tags__y__gte': 50}

    query = QueryParser({'$or': [{'price': {'$gte': 0}}, {'price': {'$lte': 54}}]})
    assert len(query.lookup_groups.children) == 2
    assert query.lookup_groups.op == 'or'
    assert query.lookup_groups.children[0].lookups == {'price__gte': 0}
    assert query.lookup_groups.children[1].lookups == {'price__lte': 54}


def test_complex_query():
    conditions = {
        '$and': {
            'price': {'$lt': 50},
            'rating': {'$gte': 1},
            'year': {'$gte': 2007},
        }
    }
    query = QueryParser(conditions)
    print(conditions)
    print(query.lookup_groups)

    # conditions = {
    #     '$and': {
    #         'price': {'$or': [{'price': {'$gte': 0}}, {'price': {'$lte': 54}}]},
    #         'rating': {'$gte': 1},
    #         'year': {'$gte': 2007, '$lte': 2010},
    #     }
    # }
    # query = QueryParser(conditions)
    # print(conditions)
    # print(query.lookup_groups)
