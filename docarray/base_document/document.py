import os
from typing import TYPE_CHECKING, Any, Optional, Type

import numpy as np
import orjson
from pydantic import BaseModel, Field, parse_obj_as
from rich.tree import Tree
from typing_inspect import is_optional_type, is_union_type

from docarray.base_document.abstract_document import AbstractDocument
from docarray.base_document.base_node import BaseNode
from docarray.base_document.io.json import orjson_dumps, orjson_dumps_and_decode
from docarray.base_document.mixins import ProtoMixin
from docarray.math.helper import minmax_normalize
from docarray.typing import ID

if TYPE_CHECKING:
    from rich.console import Console, ConsoleOptions, RenderResult
    from rich.measure import Measurement


class BaseDocument(BaseModel, ProtoMixin, AbstractDocument, BaseNode):
    """
    The base class for Document
    """

    id: ID = Field(default_factory=lambda: parse_obj_as(ID, os.urandom(16).hex()))

    class Config:
        json_loads = orjson.loads
        json_dumps = orjson_dumps_and_decode
        json_encoders = {dict: orjson_dumps}

        validate_assignment = True

    @classmethod
    def _get_field_type(cls, field: str) -> Type['BaseDocument']:
        """
        Accessing the nested python Class define in the schema. Could be useful for
        reconstruction of Document in serialization/deserilization
        :param field: name of the field
        :return:
        """
        return cls.__fields__[field].outer_type_

    def summary(self) -> None:
        """Print non-empty fields and nested structure of this Document object."""
        import rich

        t = _plot_recursion(node=self)
        rich.print(t)

    @classmethod
    def schema_summary(cls) -> None:
        """Print a summary of the Documents schema."""
        import rich

        panel = rich.panel.Panel(
            cls.get_schema(), title='Document Schema', expand=False, padding=(1, 3)
        )
        rich.print(panel)

    @classmethod
    def get_schema(cls, doc_name: Optional[str] = None) -> Tree:
        """Get Documents schema as a rich.tree.Tree object."""
        import re

        from rich.tree import Tree

        from docarray import DocumentArray

        name = cls.__name__
        tree = Tree(name) if doc_name is None else Tree(f'{doc_name}: {name}')

        for k, v in cls.__annotations__.items():

            field_type = cls._get_field_type(k)

            t = str(v).replace('[', '\[')
            t = re.sub('[a-zA-Z_]*[.]', '', t)

            if is_union_type(v) or is_optional_type(v):
                sub_tree = Tree(f'{k}: {t}')
                for arg in v.__args__:
                    if issubclass(arg, BaseDocument):
                        sub_tree.add(arg.get_schema())
                    elif issubclass(arg, DocumentArray):
                        sub_tree.add(arg.document_type.get_schema())
                tree.add(sub_tree)
            elif issubclass(field_type, BaseDocument):
                tree.add(field_type.get_schema(doc_name=k))
            elif issubclass(field_type, DocumentArray):
                field_cls = v.__name__.replace('[', '\[')
                sub_tree = Tree(f'{k}: {field_cls}')
                sub_tree.add(field_type.document_type.get_schema())
                tree.add(sub_tree)
            else:
                tree.add(f'{k}: {t}')
        return tree

    def __rich_console__(self, console, options):
        kls = self.__class__.__name__
        id_abbrv = getattr(self, 'id')[:7]
        yield f":page_facing_up: [b]{kls}" f"[/b]: [cyan]{id_abbrv} ...[cyan]"

        import torch
        from rich import box, text
        from rich.table import Table

        import docarray

        table = Table('Attribute', 'Value', width=80, box=box.ROUNDED, highlight=True)

        for k, v in self.__dict__.items():
            col_1 = f'{k}: {v.__class__.__name__}'
            if (
                isinstance(v, ID | docarray.DocumentArray | docarray.BaseDocument)
                or k.startswith('_')
                or v is None
            ):
                continue
            elif isinstance(v, str):
                col_2 = str(v)[:50]
                if len(v) > 50:
                    col_2 += f' ... (length: {len(v)})'
                table.add_row(col_1, text.Text(col_2))
            elif isinstance(v, np.ndarray | torch.Tensor):
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu().numpy()
                if v.squeeze().ndim == 1 and len(v) < 200:
                    table.add_row(col_1, ColorBoxArray(v.squeeze()))
                else:
                    table.add_row(
                        col_1,
                        text.Text(f'{type(v)} of shape {v.shape}, dtype: {v.dtype}'),
                    )
            elif isinstance(v, tuple | list):
                col_2 = ''
                for i, x in enumerate(v):
                    if len(col_2) + len(str(x)) < 50:
                        col_2 = str(v[:i])
                    else:
                        col_2 = f'{col_2[:-1]}, ...] (length: {len(v)})'
                        break
                table.add_row(col_1, text.Text(col_2))

        if table.rows:
            yield table


def _plot_recursion(node: Any, tree: Optional[Tree] = None) -> Tree:
    """
    Store node's children in rich.tree.Tree recursively.

    :param node: Node to get children from.
    :param tree: Append to this tree if not None, else use node as root.
    :return: Tree with all children.

    """
    import docarray

    tree = Tree(node) if tree is None else tree.add(node)

    if hasattr(node, '__dict__'):
        iterable_attrs = [
            k
            for k, v in node.__dict__.items()
            if isinstance(v, docarray.DocumentArray | docarray.BaseDocument)
        ]
        for attr in iterable_attrs:
            value = getattr(node, attr)
            attr_type = value.__class__.__name__
            icon = ':diamond_with_a_dot:'

            if isinstance(value, docarray.BaseDocument):
                icon = ':large_orange_diamond:'
                value = [value]

            match_tree = tree.add(f'{icon} [b]{attr}: ' f'{attr_type}[/b]')
            for i, d in enumerate(value):
                if i == 2:
                    doc_type = d.__class__.__name__
                    _plot_recursion(
                        node=f'... {len(value) - 2} more {doc_type} documents\n',
                        tree=match_tree,
                    )
                    break
                _plot_recursion(d, match_tree)

    return tree


class ColorBoxArray:
    """
    Rich representation of an array as coloured blocks.
    """

    def __init__(self, array):
        self._array = minmax_normalize(array, (0, 5))

    def __rich_console__(
        self, console: 'Console', options: 'ConsoleOptions'
    ) -> 'RenderResult':
        import colorsys

        from rich.color import Color
        from rich.segment import Segment
        from rich.style import Style

        h = 0.75
        for idx, y in enumerate(self._array):
            lightness = 0.1 + ((y / 5) * 0.7)
            r, g, b = colorsys.hls_to_rgb(h, lightness + 0.7 / 10, 1.0)
            color = Color.from_rgb(r * 255, g * 255, b * 255)
            yield Segment('▄', Style(color=color, bgcolor=color))
            if idx != 0 and idx % options.max_width == 0:
                yield Segment.line()

    def __rich_measure__(
        self, console: 'Console', options: 'ConsoleOptions'
    ) -> 'Measurement':
        from rich.measure import Measurement

        return Measurement(1, options.max_width)
