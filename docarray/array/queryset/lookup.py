import re
from functools import partial


## filter and lookup functions


def filter_items(items, *args, **kwargs):
    """Filters an iterable using lookup parameters

    :param items  : iterable
    :param args   : ``Q`` objects
    :param kwargs : lookup parameters
    :rtype        : lazy iterable (generator)

    """
    q1 = list(args) if args is not None else []
    q2 = [Q(**kwargs)] if kwargs is not None else []
    lookup_groups = q1 + q2
    pred = lambda item: all(lg.evaluate(item) for lg in lookup_groups)
    return (item for item in items if pred(item))


def lookup(key, val, doc):
    """Checks if key-val pair exists in item using various lookup types

    The lookup types are derived from the `key` and then used to check
    if the lookup holds true for the item::

        >>> lookup('request__url__exact', 'http://example.com', item)

    The above will return True if item['request']['url'] ==
    'http://example.com' else False

    :param key  : (str) that represents the field name to find
    :param val  : (mixed) object to match the value in the item against
    :param item : (dict)
    :rtype      : (boolean) True if field-val exists else False

    """
    get_key, last = dunder_partition(key)

    value = doc._get_attributes(get_key)
    if last == 'exact':
        return value == val
    elif last == 'neq':
        return value != val
    elif last == 'contains':
        val = guard_str(val)
        return iff_not_none(value, lambda y: val in y)
    elif last == 'icontains':
        val = guard_str(val)
        return iff_not_none(value, lambda y: val.lower() in y.lower())
    elif last == 'in':
        val = guard_iter(val)
        return value in val
    elif last == 'nin':
        val = guard_iter(val)
        return not (value in val)
    elif last == 'startswith':
        val = guard_str(val)
        return iff_not_none(value, lambda y: y.startswith(val))
    elif last == 'istartswith':
        val = guard_str(val)
        return iff_not_none(value, lambda y: y.lower().startswith(val.lower()))
    elif last == 'endswith':
        val = guard_str(val)
        return iff_not_none(value, lambda y: y.endswith(val))
    elif last == 'iendswith':
        val = guard_str(val)
        return iff_not_none(value, lambda y: y.lower().endswith(val.lower()))
    elif last == 'gt':
        return iff_not_none(value, lambda y: y > val)
    elif last == 'gte':
        return iff_not_none(value, lambda y: y >= val)
    elif last == 'lt':
        return iff_not_none(value, lambda y: y < val)
    elif last == 'lte':
        return iff_not_none(value, lambda y: y <= val)
    elif last == 'regex':
        return iff_not_none(value, lambda y: re.search(val, y) is not None)
    elif last == 'filter':
        val = guard_Q(val)
        result = guard_list(value)
        return len(list(filter_items(result, val))) > 0
    else:
        return value == val


## Classes to compose compound lookups (Q object)


class LookupTreeElem(object):
    """Base class for a child in the lookup expression tree"""

    def __init__(self):
        self.negate = False

    def evaluate(self, item):
        raise NotImplementedError

    def __or__(self, other):
        node = LookupNode()
        node.op = 'or'
        node.add_child(self)
        node.add_child(other)
        return node

    def __and__(self, other):
        node = LookupNode()
        node.add_child(self)
        node.add_child(other)
        return node


class LookupNode(LookupTreeElem):
    """A node (element having children) in the lookup expression tree

    Typically it's any object composed of two ``Q`` objects eg::

        >>> Q(language__neq='Ruby') | Q(framework__startswith='S')
        >>> ~Q(language__exact='PHP')

    """

    def __init__(self):
        super(LookupNode, self).__init__()
        self.children = []
        self.op = 'and'

    def add_child(self, child):
        self.children.append(child)

    def evaluate(self, item):
        """Evaluates the expression represented by the object for the item

        :param item : (dict) item
        :rtype      : (boolean) whether lookup passed or failed

        """
        results = map(lambda x: x.evaluate(item), self.children)
        result = any(results) if self.op == 'or' else all(results)
        return not result if self.negate else result

    def __invert__(self):
        newnode = LookupNode()
        for c in self.children:
            newnode.add_child(c)
        newnode.negate = not self.negate
        return newnode

    def __repr__(self):
        return f'{self.op}: [{self.children}]'


class LookupLeaf(LookupTreeElem):
    """Class for a leaf in the lookup expression tree"""

    def __init__(self, **kwargs):
        super(LookupLeaf, self).__init__()
        self.lookups = kwargs

    def evaluate(self, item):
        """Evaluates the expression represented by the object for the item

        :param item : (dict) item
        :rtype      : (boolean) whether lookup passed or failed

        """
        result = all(lookup(k, v, item) for k, v in self.lookups.items())
        return not result if self.negate else result

    def __invert__(self):
        newleaf = LookupLeaf(**self.lookups)
        newleaf.negate = not self.negate
        return newleaf

    def __repr__(self):
        return f'{self.lookups}'


# alias LookupLeaf to Q
Q = LookupLeaf


## Exceptions


class LookupyError(Exception):
    """Base exception class for all exceptions raised by lookupy"""

    pass


## utility functions


def dunder_partition(key):
    """Splits a dunderkey into 2 parts
    The first part is everything before the final double underscore
    The second part is after the final double underscore
        >>> dunder_partition('a__b__c')
        >>> ('a__b', 'c')
    :param neskey : String
    :rtype        : 2 Tuple
    """
    parts = key.rsplit('__', 1)
    return tuple(parts) if len(parts) > 1 else (parts[0], None)


def iff(precond, val, f):
    """If and only if the precond is True

    Shortcut function for precond(val) and f(val). It is mainly used
    to create partial functions for commonly required preconditions

    :param precond : (function) represents the precondition
    :param val     : (mixed) value to which the functions are applied
    :param f       : (function) the actual function

    """
    return False if not precond(val) else f(val)


iff_not_none = partial(iff, lambda x: x is not None)


def guard_type(classinfo, val):
    if not isinstance(val, classinfo):
        raise LookupyError('Value not a {classinfo}'.format(classinfo=classinfo))
    return val


guard_str = partial(guard_type, str)
guard_list = partial(guard_type, list)
guard_Q = partial(guard_type, Q)


def guard_iter(val):
    try:
        iter(val)
    except TypeError:
        raise LookupyError('Value not an iterable')
    else:
        return val
