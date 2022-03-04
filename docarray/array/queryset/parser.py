from typing import Dict, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ... import Document

from .lookup import Q, LookupNode, LookupLeaf

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


def _parse_lookups(data: Dict = {}, root_node: Optional[LookupNode] = None):
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(root_node, LookupLeaf):
                root = LookupNode()
                root.add_child(root_node)
                root_node = root

            if key in LOGICAL_OPERATORS:
                node = LookupNode(op=key[1:])
                node = _parse_lookups(value, root_node=node)

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
                        node = LookupNode(op=op[1:])
                        node = _parse_lookups(val, root_node=node)
                    elif op in SUPPORTED_OPERATORS:
                        node = Q(**{f'{key}__{SUPPORTED_OPERATORS[op]}': val})
                    else:
                        raise ValueError(
                            f'The operator {op} is not supported yet, please double check the given filters!'
                        )

                else:
                    node = LookupNode()
                    for op, val in items:
                        _node = _parse_lookups({key: {op: val}})
                        node.add_child(_node)

            if root_node and node:
                root_node.add_child(node)
            elif node:
                root_node = node

    elif isinstance(data, list):
        for d in data:
            node = _parse_lookups(d)
            if root_node and node:
                root_node.add_child(node)
            elif node:
                root_node = node
    else:
        raise ValueError(f'The query is illegal: {data}')

    return root_node


class QueryParser:
    """A class to parse dict condition to lookup query."""

    def __init__(self, conditions: Dict = {}):
        self.conditions = conditions
        self.lookup_groups = _parse_lookups(self.conditions)

    def evaluate(self, doc: 'Document'):
        return self.lookup_groups.evaluate(doc) if self.lookup_groups else True

    def __call__(self, doc: 'Document'):
        return self.evaluate(doc)
