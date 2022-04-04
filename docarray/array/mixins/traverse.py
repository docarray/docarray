import itertools
import re
from typing import (
    Iterable,
    TYPE_CHECKING,
    Optional,
    Callable,
    Dict,
    List,
)

if TYPE_CHECKING:
    from ... import DocumentArray, Document
    from ...typing import T

ATTRIBUTES_SEPARATOR = ','
PATHS_SEPARATOR = ','

SLICE_BASE = r'[-\d:]+'
WRAPPED_SLICE_BASE = r'\[[-\d:]+\]'

SLICE = rf'({SLICE_BASE}|{WRAPPED_SLICE_BASE})?'
SLICE_TAGGED = rf'(?P<slice>{SLICE})'

ATTRIBUTE_NAME = r'[a-zA-Z][a-zA-Z0-9]*'

# accepts both syntaxes: '.[att]' or '.att'
# However, this makes the grammar ambiguous. E.g:
# 'r.attr' should it be parsed into tokens 'r', '.', 'attr' or 'r', '.', 'att', 'r' ?
ATTRIBUTE = rf'\.(\[({ATTRIBUTE_NAME}({ATTRIBUTES_SEPARATOR}{ATTRIBUTE_NAME})*)\]|{ATTRIBUTE_NAME})'
ATTRIBUTE_TAGGED = rf'\.(\[(?P<attributes>{ATTRIBUTE_NAME}({ATTRIBUTES_SEPARATOR}{ATTRIBUTE_NAME})*)\]|(?P<attribute>{ATTRIBUTE_NAME}))'

SELECTOR = rf'(r|c|m|{ATTRIBUTE})'
SELECTOR_TAGGED = rf'(?P<selector>r|c|m|{ATTRIBUTE_TAGGED})'

REMAINDER = rf'({SELECTOR}{SLICE})*'
REMAINDER_TAGGED = rf'(?P<remainder>({SELECTOR}{SLICE})*)'

TRAVERSAL_PATH = rf'{SELECTOR}{SLICE}{REMAINDER}'
TRAVERSAL_PATH_TAGGED = rf'(?P<path>{SELECTOR_TAGGED}{SLICE_TAGGED}){REMAINDER_TAGGED}'

PATHS_REMAINDER_TAGGED = rf'(?P<paths_remainder>({PATHS_SEPARATOR}{TRAVERSAL_PATH})*)'

TRAVERSAL_PATH_LIST_TAGGED = (
    rf'^(?P<traversal_path>{TRAVERSAL_PATH}){PATHS_REMAINDER_TAGGED}$'
)

ATTRIBUTE_REGEX = re.compile(rf'^{ATTRIBUTE}$')
TRAVERSAL_PATH_REGEX = re.compile(rf'^{TRAVERSAL_PATH_TAGGED}$')
TRAVERSAL_PATH_LIST_REGEX = re.compile(TRAVERSAL_PATH_LIST_TAGGED)


def _re_traversal_path_split(path: str) -> List[str]:
    res = []
    remainder = path
    while True:
        m = TRAVERSAL_PATH_LIST_REGEX.match(remainder)
        if not m:
            raise ValueError(
                f'`path`:{path} is invalid, please refer to https://docarray.jina.ai/fundamentals/documentarray/access-elements/#index-by-nested-structure'
            )
        group_dict = m.groupdict()
        current, remainder = group_dict['traversal_path'], group_dict['paths_remainder']
        res.append(current)
        if not remainder:
            break
        else:
            remainder = remainder[1:]

    return res


class TraverseMixin:
    """
    A mixin used for traversing :class:`DocumentArray`.
    """

    def traverse(
        self: 'T',
        traversal_paths: str,
        filter_fn: Optional[Callable[['Document'], bool]] = None,
    ) -> Iterable['T']:
        """
        Return an Iterator of :class:``TraversableSequence`` of the leaves when applying the traversal_paths.
        Each :class:``TraversableSequence`` is either the root Documents, a ChunkArray or a MatchArray.

        :param traversal_paths: a comma-separated string that represents the traversal path
        :param filter_fn: function to filter docs during traversal
        :yield: :class:``TraversableSequence`` of the leaves when applying the traversal_paths.

        Example on ``traversal_paths``:

            - `r`: docs in this TraversableSequence
            - `m`: all match-documents at adjacency 1
            - `c`: all child-documents at granularity 1
            - `r.[attribute]`: access attribute of a multi modal document
            - `cc`: all child-documents at granularity 2
            - `mm`: all match-documents at adjacency 2
            - `cm`: all match-document at adjacency 1 and granularity 1
            - `r,c`: docs in this TraversableSequence and all child-documents at granularity 1
            - `r[start:end]`: access sub document array using slice

        """
        traversal_paths = re.sub(r'\s+', '', traversal_paths)
        for p in _re_traversal_path_split(traversal_paths):
            yield from self._traverse(self, p, filter_fn=filter_fn)

    @staticmethod
    def _traverse(
        docs: 'T',
        path: str,
        filter_fn: Optional[Callable[['Document'], bool]] = None,
    ):
        path = re.sub(r'\s+', '', path)
        if path:
            group_dict = _parse_path_string(path)
            cur_loc = group_dict['selector']
            cur_slice = group_dict['slice']
            remainder = group_dict['remainder']

            if cur_loc == 'r':
                yield from TraverseMixin._traverse(
                    docs[cur_slice], remainder, filter_fn=filter_fn
                )
            elif cur_loc == 'm':
                for d in docs:
                    yield from TraverseMixin._traverse(
                        d.matches[cur_slice], remainder, filter_fn=filter_fn
                    )
            elif cur_loc == 'c':
                for d in docs:
                    yield from TraverseMixin._traverse(
                        d.chunks[cur_slice], remainder, filter_fn=filter_fn
                    )
            elif ATTRIBUTE_REGEX.match(cur_loc):
                for d in docs:
                    for attribute in group_dict['attributes']:
                        yield from TraverseMixin._traverse(
                            d.get_multi_modal_attribute(attribute)[cur_slice],
                            remainder,
                            filter_fn=filter_fn,
                        )
            else:
                raise ValueError(
                    f'`path`:{path} is invalid, please refer to https://docarray.jina.ai/fundamentals/documentarray/access-elements/#index-by-nested-structure'
                )
        elif filter_fn is None:
            yield docs
        else:
            from .. import DocumentArray

            yield DocumentArray(list(filter(filter_fn, docs)))

    def traverse_flat_per_path(
        self,
        traversal_paths: str,
        filter_fn: Optional[Callable[['Document'], bool]] = None,
    ):
        """
        Returns a flattened :class:``TraversableSequence`` per path in ``traversal_paths``
        with all Documents, that are reached by the path.

        :param traversal_paths: a comma-separated string that represents the traversal path
        :param filter_fn: function to filter docs during traversal
        :yield: :class:``TraversableSequence`` containing the document of all leaves per path.
        """
        traversal_paths = re.sub(r'\s+', '', traversal_paths)
        for p in _re_traversal_path_split(traversal_paths):
            yield self._flatten(self._traverse(self, p, filter_fn=filter_fn))

    def traverse_flat(
        self,
        traversal_paths: str,
        filter_fn: Optional[Callable[['Document'], bool]] = None,
    ) -> 'DocumentArray':
        """
        Returns a single flattened :class:``TraversableSequence`` with all Documents, that are reached
        via the ``traversal_paths``.

        .. warning::
            When defining the ``traversal_paths`` with multiple paths, the returned
            :class:``Documents`` are determined at once and not on the fly. This is a different
            behavior then in :method:``traverse`` and :method:``traverse_flattened_per_path``!

        :param traversal_paths: a list of string that represents the traversal path
        :param filter_fn: function to filter docs during traversal
        :return: a single :class:``TraversableSequence`` containing the document of all leaves when applying the traversal_paths.
        """
        if traversal_paths == 'r' and filter_fn is None:
            return self

        leaves = self.traverse(traversal_paths, filter_fn=filter_fn)
        return self._flatten(leaves)

    def flatten(self) -> 'DocumentArray':
        """Flatten all nested chunks and matches into one :class:`DocumentArray`.

        .. note::
            Flatten an already flattened DocumentArray will have no effect.

        :return: a flattened :class:`DocumentArray` object.
        """
        from .. import DocumentArray

        if hasattr(self, '_flattened') and getattr(self, '_flattened'):
            return self

        visited = set()

        def _yield_all():
            for d in self:
                yield from _yield_nest(d)

        def _yield_nest(doc: 'Document'):
            if doc.id not in visited:
                for d in doc.chunks:
                    yield from _yield_nest(d)
                for m in doc.matches:
                    yield from _yield_nest(m)
                visited.add(doc.id)

            yield doc

        da = DocumentArray(_yield_all())
        da._flattened = True
        return da

    @staticmethod
    def _flatten(sequence) -> 'DocumentArray':
        from ... import DocumentArray

        return DocumentArray(list(itertools.chain.from_iterable(sequence)))


def _parse_path_string(p: str) -> Dict[str, str]:
    g = TRAVERSAL_PATH_REGEX.match(p)
    group_dict = g.groupdict()
    group_dict['remainder'] = group_dict.get('remainder') or ''
    group_dict['slice'] = _parse_slice(group_dict.get('slice') or ':')
    if group_dict.get('attributes'):
        group_dict['attributes'] = group_dict['attributes'].split(ATTRIBUTES_SEPARATOR)
    elif group_dict.get('attribute'):
        group_dict['attributes'] = [group_dict.get('attribute')]

    return group_dict


def _parse_slice(value):
    """
    Parses a `slice()` from string, like `start:stop:step`.
    """
    if re.match(WRAPPED_SLICE_BASE, value):
        value = value[1:-1]

    if value:
        parts = value.split(':')
        if len(parts) == 1:
            # slice(stop)
            parts = [parts[0], str(int(parts[0]) + 1)]
        # else: slice(start, stop[, step])
    else:
        # slice()
        parts = []
    return slice(*[int(p) if p else None for p in parts])
