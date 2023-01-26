from typing import TYPE_CHECKING, List

from docarray.typing.tensor.abstract_tensor import AbstractTensor

if TYPE_CHECKING:
    from docarray.array import DocumentArrayStacked
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
            stacked_fields = self._get_stacked_fields(da=self.da)
            for field in stacked_fields:
                da = self.da
                for attr in field.split('.'):
                    da = getattr(da, attr)

                if isinstance(da, AbstractTensor):
                    col_1 = f'  • {field}:'
                    comp_be = da.get_comp_backend()
                    cls_name = da.__class__.__name__
                    if comp_be.isnan(da).all():
                        col_2 = f'None ({cls_name})'
                    else:
                        col_2 = (
                            f'{cls_name} of shape {comp_be.shape(da)}, '
                            f'dtype: {comp_be.shape(da)}'
                        )
                    table.add_row(col_1, col_2)

        Console().print(Panel(table, title='DocumentArray Summary', expand=False))
        self.da.document_type.schema_summary()

    @staticmethod
    def _get_stacked_fields(da: 'DocumentArrayStacked') -> List[str]:
        """
        Returns a list of field names that are stacked of a DocumentArrayStacked
        instance, i.e. all the fields that are of type AbstractTensor. Nested field
        paths are dot separated.
        """
        from docarray.array import DocumentArrayStacked

        fields = []
        for field_name, value in da._columns.items():
            if isinstance(value, AbstractTensor):
                fields.append(field_name)
            elif isinstance(value, DocumentArrayStacked):
                fields.extend(
                    [
                        f'{field_name}.{x}'
                        for x in DocumentArraySummary._get_stacked_fields(da=value)
                    ]
                )

        return fields
