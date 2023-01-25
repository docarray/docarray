from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from docarray.array.abstract_array import AnyDocumentArray


class DocumentArraySummary:
    def __init__(
        self,
        da: Optional['AnyDocumentArray'] = None,
    ):
        self.da = da

    def summary(self) -> None:
        """
        Print a summary of this DocumentArray object and a summary of the schema of its
        Document type.
        """
        from rich import box
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        table = Table(box=box.SIMPLE, highlight=True)
        table.show_header = False
        table.add_row('Type', self.da.__class__.__name__)
        table.add_row('Length', str(len(self.da)))

        Console().print(Panel(table, title='DocumentArray Summary', expand=False))
        self.da.document_type.schema_summary()
