from typing import Any, Optional, Type, Union

from rich.highlighter import RegexHighlighter
from rich.theme import Theme
from rich.tree import Tree
from typing_extensions import TYPE_CHECKING
from typing_inspect import is_optional_type, is_union_type

from docarray.base_document.abstract_document import AbstractDocument
from docarray.typing import ID
from docarray.typing.tensor.abstract_tensor import AbstractTensor

if TYPE_CHECKING:
    from rich.console import Console, ConsoleOptions, RenderResult
    from rich.measure import Measurement


class DocumentSummary:
    table_width: int = 80

    def __init__(
        self,
        doc: Optional['AbstractDocument'] = None,
        doc_cls: Optional[Type['AbstractDocument']] = None,
    ):
        self.doc = doc
        self.doc_cls = doc_cls

    def summary(self) -> None:
        """Print non-empty fields and nested structure of this Document object."""
        import rich

        t = self._plot_recursion(node=self)
        rich.print(t)

    @staticmethod
    def schema_summary(cls: Type['AbstractDocument']) -> None:
        """Print a summary of the Documents schema."""
        from rich.console import Console
        from rich.panel import Panel

        panel = Panel(
            DocumentSummary._get_schema(cls),
            title='Document Schema',
            expand=False,
            padding=(1, 3),
        )
        highlighter = SchemaHighlighter()

        console = Console(highlighter=highlighter, theme=highlighter.theme)
        console.print(panel)

    @staticmethod
    def _get_schema(
        cls: Type['AbstractDocument'], doc_name: Optional[str] = None
    ) -> Tree:
        """Get Documents schema as a rich.tree.Tree object."""
        import re

        from rich.tree import Tree

        from docarray import BaseDocument, DocumentArray

        root = cls.__name__ if doc_name is None else f'{doc_name}: {cls.__name__}'
        tree = Tree(root, highlight=True)

        for field_name, value in cls.__fields__.items():
            if field_name != 'id':
                field_type = value.type_
                if not value.required:
                    field_type = Optional[field_type]

                field_cls = str(field_type).replace('[', '\[')
                field_cls = re.sub('<class \'|\'>|[a-zA-Z_]*[.]', '', field_cls)

                node_name = f'{field_name}: {field_cls}'

                if is_union_type(field_type) or is_optional_type(field_type):
                    sub_tree = Tree(node_name, highlight=True)
                    for arg in field_type.__args__:
                        if issubclass(arg, BaseDocument):
                            sub_tree.add(DocumentSummary._get_schema(cls=arg))
                        elif issubclass(arg, DocumentArray):
                            sub_tree.add(
                                DocumentSummary._get_schema(cls=arg.document_type)
                            )
                    tree.add(sub_tree)

                elif issubclass(field_type, BaseDocument):
                    tree.add(
                        DocumentSummary._get_schema(cls=field_type, doc_name=field_name)
                    )

                elif issubclass(field_type, DocumentArray):
                    sub_tree = Tree(node_name, highlight=True)
                    sub_tree.add(
                        DocumentSummary._get_schema(cls=field_type.document_type)
                    )
                    tree.add(sub_tree)

                else:
                    tree.add(node_name)

        return tree

    def __rich_console__(self, console, options):
        kls = self.doc.__class__.__name__
        id_abbrv = getattr(self.doc, 'id')[:7]
        yield f":page_facing_up: [b]{kls}" f"[/b]: [cyan]{id_abbrv} ...[cyan]"

        from rich import box, text
        from rich.table import Table

        from docarray import BaseDocument, DocumentArray

        table = Table(
            'Attribute',
            'Value',
            width=self.table_width,
            box=box.ROUNDED,
            highlight=True,
        )

        for field_name, value in self.doc.__dict__.items():
            col_1 = f'{field_name}: {value.__class__.__name__}'
            if (
                isinstance(value, (ID, DocumentArray, BaseDocument))
                or field_name.startswith('_')
                or value is None
            ):
                continue
            elif isinstance(value, str):
                col_2 = str(value)[:50]
                if len(value) > 50:
                    col_2 += f' ... (length: {len(value)})'
                table.add_row(col_1, text.Text(col_2))
            elif isinstance(value, AbstractTensor):
                comp = value.get_comp_backend()
                v_squeezed = comp.squeeze(comp.detach(value))
                if comp.n_dim(v_squeezed) == 1 and comp.shape(v_squeezed)[0] < 200:
                    table.add_row(col_1, ColorBoxArray(v_squeezed))
                else:
                    table.add_row(
                        col_1,
                        text.Text(f'{type(value)} of shape {comp.shape(value)}'),
                    )
            elif isinstance(value, (tuple, list)):
                col_2 = ''
                for i, x in enumerate(value):
                    if len(col_2) + len(str(x)) < 50:
                        col_2 = str(value[:i])
                    else:
                        col_2 = f'{col_2[:-1]}, ...] (length: {len(value)})'
                        break
                table.add_row(col_1, text.Text(col_2))

        if table.rows:
            yield table

    @staticmethod
    def _plot_recursion(
        node: Union['DocumentSummary', Any], tree: Optional[Tree] = None
    ) -> Tree:
        """
        Store node's children in rich.tree.Tree recursively.

        :param node: Node to get children from.
        :param tree: Append to this tree if not None, else use node as root.
        :return: Tree with all children.

        """
        from docarray import BaseDocument, DocumentArray

        tree = Tree(node) if tree is None else tree.add(node)  # type: ignore

        if hasattr(node, '__dict__'):
            nested_attrs = [
                k
                for k, v in node.doc.__dict__.items()
                if isinstance(v, (DocumentArray, BaseDocument))
            ]
            for attr in nested_attrs:
                value = getattr(node.doc, attr)
                attr_type = value.__class__.__name__
                icon = ':diamond_with_a_dot:'

                if isinstance(value, BaseDocument):
                    icon = ':large_orange_diamond:'
                    value = [value]

                match_tree = tree.add(f'{icon} [b]{attr}: ' f'{attr_type}[/b]')
                max_show = 2
                for i, d in enumerate(value):
                    if i == max_show:
                        doc_type = d.__class__.__name__
                        DocumentSummary._plot_recursion(
                            f'... {len(value) - max_show} more {doc_type} documents\n',
                            tree=match_tree,
                        )
                        break
                    DocumentSummary._plot_recursion(DocumentSummary(doc=d), match_tree)

        return tree


class ColorBoxArray:
    """
    Rich representation of an array as coloured blocks.
    """

    def __init__(self, array: AbstractTensor):
        comp_be = array.get_comp_backend()
        self._array = comp_be.minmax_normalize(comp_be.detach(array), (0, 5))

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


class SchemaHighlighter(RegexHighlighter):
    """Highlighter to apply colors to a Document's schema tree."""

    highlights = [
        r"(?P<class>^[A-Z][a-zA-Z]*)",
        r"(?P<attr>^.*(?=:))",
        r"(?P<attr_type>(?<=:).*$)",
        r"(?P<other_chars>[\[\],:])",
    ]

    theme = Theme(
        {
            "class": "orange3",
            "attr": "green4",
            "attr_type": "medium_purple3",
            "other_chars": "black",
        }
    )
