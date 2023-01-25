from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docarray.array.abstract_array import AnyDocumentArray


class DocumentArraySummary:
    def __init__(self, da: 'AnyDocumentArray'):
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

        from docarray.array import DocumentArrayStacked

        table = Table(box=box.SIMPLE, highlight=True)
        table.show_header = False
        table.add_row('Type', self.da.__class__.__name__)
        table.add_row('Length', str(len(self.da)))

        if isinstance(self.da, DocumentArrayStacked):
            table.add_section()
            table.add_row('Stacked columns:')
            for field_name, value in self.da._columns.items():
                shape = value.get_comp_backend().shape(value)
                table.add_row(
                    f'  • {field_name}:',
                    f'{value.__class__.__name__} of shape {shape}',
                )

        Console().print(Panel(table, title='DocumentArray Summary', expand=False))
        self.da.document_type.schema_summary()
