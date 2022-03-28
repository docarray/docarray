"""

Originally from https://github.com/naiquevin/lookupy

The library is provided as-is under the MIT License

Copyright (c) 2013 Vineet Naik (naikvin@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ... import Document
import re
from functools import partial

PLACEHOLDER_PATTERN = re.compile(r'\{\s*([a-zA-Z0-9_]*)\s*}')


def lookup(key, val, doc: 'Document') -> bool:
    """Checks if key-val pair exists in doc using various lookup types

    The lookup types are derived from the `key` and then used to check
    if the lookup holds true for the document::

        >>> lookup('text__exact', 'hello', doc)

    The above will return True if doc.text == 'hello' else False. And

        >>> lookup('text_exact', '{tags__name}', doc)

    will return True if doc.text == doc.tags['name'] else False

    :param key  : the field name to find
    :param val  : object to match the value in the document against
    :param doc : the document to match
    """
    get_key, last = dunder_partition(key)

    if isinstance(val, str) and val.startswith('{'):
        r = PLACEHOLDER_PATTERN.findall(val)
        if r and len(r) == 1:
            val = doc._get_attributes(r[0])
        else:
            raise ValueError(f'The placeholder "{val}" is illegal')

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
    elif last == 'size':
        return iff_not_none(value, lambda y: len(y) == val)
    elif last == 'exists':
        if not isinstance(val, bool):
            raise ValueError(
                '$exists operator can only accept True/False as value for comparison'
            )

        if '__' in get_key:
            is_empty = False
            try:
                is_empty = not value
            except:
                # ndarray-like will end up here
                pass

            return is_empty != val
        else:
            return (get_key in doc.non_empty_fields) == val
    else:
        # return value == val
        raise ValueError(
            f'The given compare operator "{last}" (derived from "{key}") is not supported'
        )


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

    def __init__(self, op: str = 'and', negate: bool = False):
        super(LookupNode, self).__init__()
        self.children = []
        self.op = op
        self.negate = negate

    def add_child(self, child):
        self.children.append(child)

    def evaluate(self, doc: 'Document'):
        """Evaluates the expression represented by the object for the document

        :param doc : the document to match
        :return: returns true if lookup passed
        """
        results = map(lambda x: x.evaluate(doc), self.children)
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

    def evaluate(self, doc: 'Document'):
        """Evaluates the expression represented by the object for the document

        :param doc : the document to match
        :return: returns true if lookup passed
        """
        result = all(lookup(k, v, doc) for k, v in self.lookups.items())
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
