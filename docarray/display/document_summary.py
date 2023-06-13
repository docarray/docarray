from typing import Any, Optional, Type, Union

from rich.highlighter import RegexHighlighter
from rich.theme import Theme
from rich.tree import Tree
from typing_extensions import TYPE_CHECKING
from typing_inspect import is_optional_type, is_union_type

from docarray.base_doc.doc import BaseDoc
from docarray.display.tensor_display import TensorDisplay
from docarray.typing import ID
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.utils._internal._typing import safe_issubclass

if TYPE_CHECKING:
    from rich.console import Console, ConsoleOptions, RenderResult


class DocumentSummary:
    table_min_width: int = 40

    def __init__(
        self,
        doc: Optional['BaseDoc'] = None,
    ):
        self.doc = doc

    def summary(self) -> None:
        """Print non-empty fields and nested structure of this Document object."""
        import rich

        t = self._plot_recursion(node=self)
        rich.print(t)

    @staticmethod
    def schema_summary(cls: Type['BaseDoc']) -> None:
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
    def _get_schema(cls: Type['BaseDoc'], doc_name: Optional[str] = None) -> Tree:
        """Get Documents schema as a rich.tree.Tree object."""
        import re

        from rich.tree import Tree

        from docarray import BaseDoc, DocList

        root = cls.__name__ if doc_name is None else f'{doc_name}: {cls.__name__}'
        tree = Tree(root, highlight=True)

        for field_name, value in cls.__fields__.items():
            if field_name != 'id':
                field_type = value.annotation
                field_cls = str(field_type).replace('[', '\[')
                field_cls = re.sub('<class \'|\'>|[a-zA-Z_]*[.]', '', field_cls)

                node_name = f'{field_name}: {field_cls}'

                if is_union_type(field_type) or is_optional_type(field_type):
                    sub_tree = Tree(node_name, highlight=True)
                    for arg in field_type.__args__:
                        if safe_issubclass(arg, BaseDoc):
                            sub_tree.add(DocumentSummary._get_schema(cls=arg))
                        elif safe_issubclass(arg, DocList):
                            sub_tree.add(DocumentSummary._get_schema(cls=arg.doc_type))
                    tree.add(sub_tree)

                elif safe_issubclass(field_type, BaseDoc):
                    tree.add(
                        DocumentSummary._get_schema(cls=field_type, doc_name=field_name)
                    )

                elif safe_issubclass(field_type, DocList):
                    sub_tree = Tree(node_name, highlight=True)
                    sub_tree.add(DocumentSummary._get_schema(cls=field_type.doc_type))
                    tree.add(sub_tree)

                else:
                    tree.add(node_name)

        return tree

    def __rich_console__(
        self, console: 'Console', options: 'ConsoleOptions'
    ) -> 'RenderResult':
        kls = self.doc.__class__.__name__
        doc_id = getattr(self.doc, 'id')
        if doc_id is not None:
            yield f':page_facing_up: [b]{kls} [/b]: [cyan]{doc_id[:7]} ...[cyan]'
        else:
            yield f':page_facing_up: [b]{kls} [/b]'

        from rich import box, text
        from rich.table import Table

        from docarray import BaseDoc, DocList

        table = Table(
            'Attribute',
            'Value',
            min_width=self.table_min_width,
            box=box.ROUNDED,
            highlight=True,
        )

        for field_name, value in self.doc.__dict__.items():
            col_1 = f'{field_name}: {value.__class__.__name__}'
            if (
                isinstance(value, (ID, DocList, BaseDoc))
                or field_name.startswith('_')
                or value is None
            ):
                continue
            elif isinstance(value, (str, bytes)):
                col_2 = str(value)[:50]
                if len(value) > 50:
                    col_2 += f' ... (length: {len(value)})'
                table.add_row(col_1, text.Text(col_2))
            elif isinstance(value, AbstractTensor):
                table.add_row(col_1, TensorDisplay(tensor=value))
            elif isinstance(value, (tuple, list, set, frozenset)):
                value_list = list(value)
                col_2 = ''
                for i, x in enumerate(value_list):
                    if len(col_2) + len(str(x)) < 50:
                        col_2 = str(value_list[: i + 1])
                    else:
                        col_2 = f'{col_2[:-1]}, ...] (length: {len(value_list)})'
                        break

                if type(value) == tuple:
                    col_2 = col_2.replace('[', '(', 1).replace(']', ')', -1)
                if type(value) == set or type(value) == frozenset:
                    col_2 = col_2.replace('[', '{', 1).replace(']', '}', -1)

                table.add_row(col_1, text.Text(col_2))
            elif isinstance(value, dict):
                col_2 = f'{value}'
                if len(col_2) > 50:
                    col_2 = f'{col_2[: 50]}' + ' ... } ' + f'(length: {len(value)})'
                table.add_row(col_1, text.Text(col_2))
            else:
                col_2 = f'{value}'
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
        from docarray import BaseDoc, DocList

        tree = Tree(node) if tree is None else tree.add(node)  # type: ignore

        if hasattr(node, '__dict__'):
            nested_attrs = [
                k
                for k, v in node.doc.__dict__.items()
                if isinstance(v, (DocList, BaseDoc))
            ]
            for attr in nested_attrs:
                value = getattr(node.doc, attr)
                attr_type = value.__class__.__name__
                icon = ':diamond_with_a_dot:'

                if isinstance(value, BaseDoc):
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


class SchemaHighlighter(RegexHighlighter):
    """Highlighter to apply colors to a Document's schema tree."""

    highlights = [
        r'(?P<class>^[A-Z][a-zA-Z]*)',
        r'(?P<attr>^.*(?=:))',
        r'(?P<attr_type>(?<=:).*$)',
        r'(?P<union_or_opt>Union|Optional)',
        r'(?P<other_chars>[\[\],:])',
    ]

    theme = Theme(
        {
            'class': 'orange3',
            'attr': 'green4',
            'attr_type': 'medium_orchid',
            'union_or_opt': 'medium_purple4',
            'other_chars': 'black',
        }
    )
