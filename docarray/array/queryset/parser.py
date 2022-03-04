from typing import Dict, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from ... import Document

from .lookup import Q

LOGICAL_OPERATORS = {'$and': lambda x, y: x & y, '$or': lambda x, y: x | y}

COMPARISON_OPERATORS = {
    '$lt': 'lt',
    '$gt': 'gt',
    '$lte': 'lte',
    '$gte': 'gte',
    '$eq': 'exact',
    '$neq': 'neq',
}

MEMBERSHIP_OPERATORS = {'$in': 'in', '$nin': 'nin'}

REGEX_OPERATORS = {'$regex': 'regex'}

SUPPORTED_OPERATORS = {
    **COMPARISON_OPERATORS,
    **MEMBERSHIP_OPERATORS,
    **REGEX_OPERATORS,
}


def _parse_lookups(data: Dict = {}, logic_op: Callable = LOGICAL_OPERATORS['$and']):
    lookup_groups = None
    if isinstance(data, dict):
        for key, value in data.items():
            if key in LOGICAL_OPERATORS:
                _lookups = _parse_lookups(value, logic_op=LOGICAL_OPERATORS[key])
                if lookup_groups is None:
                    lookup_groups = _lookups
                else:
                    lookup_groups = LOGICAL_OPERATORS[key](lookup_groups, _lookups)
            elif key.startswith('$'):
                raise ValueError(
                    f'The operator {key} is not supported yet, please double check the given filters!'
                )
            else:
                items = list(value.items())
                if len(items) == 0:
                    raise ValueError(f'The query is illegal: {data}')

                elif len(items) == 1:
                    op, val = items[0]
                    if op in LOGICAL_OPERATORS:
                        _lookups = _parse_lookups(val, logic_op=LOGICAL_OPERATORS[op])
                    elif op in SUPPORTED_OPERATORS:
                        _lookups = Q(**{f'{key}__{SUPPORTED_OPERATORS[op]}': val})
                    else:
                        raise ValueError(
                            f'The operator {op} is not supported yet, please double check the given filters!'
                        )
                    if lookup_groups is None:
                        lookup_groups = _lookups
                    else:
                        lookup_groups = logic_op(lookup_groups, _lookups)
                else:
                    for op, val in items:
                        _lookups = _parse_lookups({key: {op: val}})
                        print(f'===> inner: {_lookups}')
                        if lookup_groups is None:
                            lookup_groups = _lookups
                        else:
                            # lookup_groups &= _lookups
                            # lookup_groups = logic_op(lookup_groups, _lookups)
                            lookup_groups.add_child(_lookups)
    elif isinstance(data, list):
        for d in data:
            _lookups = _parse_lookups(d)
            if lookup_groups is None:
                lookup_groups = _lookups
            else:
                lookup_groups = logic_op(lookup_groups, _lookups)
    else:
        raise ValueError(f'The query is illegal: {data}')

    return lookup_groups


class QueryParser:
    """A class to parse dict condition to lookup query."""

    def __init__(self, conditions: Dict = {}):
        self.conditions = conditions
        self.lookup_groups = _parse_lookups(self.conditions)

    def __call__(self, doc: 'Document'):
        return self.lookup_groups.evaluate(doc) if self.lookup_groups else True
