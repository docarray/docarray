from typing import Any, Dict, List, Optional, Union

from docarray.utils._internal.query_language.lookup import (
    LookupLeaf,
    LookupNode,
    LookupTreeElem,
    Q,
)

LOGICAL_OPERATORS: Dict[str, Union[str, bool]] = {
    '$and': 'and',
    '$or': 'or',
    '$not': True,
}

COMPARISON_OPERATORS = {
    '$lt': 'lt',
    '$gt': 'gt',
    '$lte': 'lte',
    '$gte': 'gte',
    '$eq': 'exact',
    '$neq': 'neq',
    '$exists': 'exists',
}

REGEX_OPERATORS = {'$regex': 'regex'}

ARRAY_OPERATORS = {'$size': 'size'}

MEMBERSHIP_OPERATORS = {'$in': 'in', '$nin': 'nin'}

SUPPORTED_OPERATORS = {
    **COMPARISON_OPERATORS,
    **ARRAY_OPERATORS,
    **REGEX_OPERATORS,
    **MEMBERSHIP_OPERATORS,
}


def _parse_lookups(
    data: Union[Dict, List] = {}, root_node: Optional[LookupTreeElem] = None
) -> Optional[LookupTreeElem]:
    if isinstance(data, dict):
        for key, value in data.items():
            node: Optional[LookupTreeElem] = None
            if isinstance(root_node, LookupLeaf):
                root = LookupNode()
                root.add_child(root_node)
                root_node = root

            if key in LOGICAL_OPERATORS:
                if key == '$not':
                    node = LookupNode(negate=True)
                else:
                    node = LookupNode(op=LOGICAL_OPERATORS[key])
                node = _parse_lookups(value, root_node=node)

            elif key.startswith('$'):
                raise ValueError(
                    f'The operator {key} is not supported yet,'
                    f' please double check the given filters!'
                )
            else:
                if not value or not isinstance(value, dict):
                    raise ValueError(
                        '''Not a valid query. It should follow the format:
                    { <field1>: { <operator1>: <value1> }, ... } 
                    '''
                    )

                items = list(value.items())
                if len(items) == 1:
                    op, val = items[0]
                    if op in LOGICAL_OPERATORS:
                        if op == '$not':
                            node = LookupNode(negate=True)
                        else:
                            node = LookupNode(op=LOGICAL_OPERATORS[op])
                        node = _parse_lookups(val, root_node=node)
                    elif op in SUPPORTED_OPERATORS:
                        node = Q(**{f'{key}.{SUPPORTED_OPERATORS[op]}': val})
                    else:
                        raise ValueError(
                            f'The operator {op} is not supported yet, '
                            f'please double check the given filters!'
                        )

                else:
                    node = LookupNode()
                    for op, val in items:
                        _node = _parse_lookups({key: {op: val}})
                        node.add_child(_node)

            if root_node and node:
                if isinstance(root_node, LookupNode):
                    root_node.add_child(node)
            elif node:
                root_node = node

    elif isinstance(data, list):
        for d in data:
            node = _parse_lookups(d)
            if root_node and node:
                if isinstance(root_node, LookupNode):
                    root_node.add_child(node)
            elif node:
                root_node = node
    else:
        raise ValueError(f'The query is illegal: `{data}`')

    return root_node


class QueryParser:
    """A class to parse dict condition to lookup query."""

    def __init__(self, conditions: Union[Dict, List] = {}):
        self.conditions = conditions
        self.lookup_groups = _parse_lookups(self.conditions)

    def evaluate(self, doc: Any) -> bool:
        return self.lookup_groups.evaluate(doc) if self.lookup_groups else True

    def __call__(self, doc: Any) -> bool:
        return self.evaluate(doc)
