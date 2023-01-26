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
        table.add_row('Length', str(len(self.da)), end_section=True)

        if isinstance(self.da, DocumentArrayStacked):
            table.add_row('Stacked columns:')
            stacked_fields = self._get_stacked_fields(da=self.da)
            for field_name in stacked_fields:
                val = self.da
                for attr in field_name.split('.'):
                    val = getattr(val, attr)

                if isinstance(val, AbstractTensor):
                    comp_be = val.get_comp_backend()
                    if comp_be.isnan(val).all():
                        col_2 = f'None ({val.__class__.__name__})'
                    else:
                        col_2 = (
                            f'{val.__class__.__name__} of shape {comp_be.shape(val)}'
                            f', dtype: {comp_be.dtype(val)}'
                        )
                        if comp_be.device(val):
                            col_2 += f', device: {comp_be.device(val)}'

                    table.add_row(f'  • {field_name}:', col_2)

        Console().print(Panel(table, title='DocumentArray Summary', expand=False))
        self.da.document_type.schema_summary()

    @staticmethod
    def _get_stacked_fields(da: 'DocumentArrayStacked') -> List[str]:
        """
        Return a list of the field names of a DocumentArrayStacked instance that are
        stacked, i.e. all the fields that are of type AbstractTensor. Nested field
        paths are separated by dot, such as: 'attr.nested_attr'.
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
