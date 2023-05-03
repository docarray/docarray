from typing import TYPE_CHECKING, List

from docarray.typing.tensor.abstract_tensor import AbstractTensor

if TYPE_CHECKING:
    from docarray.array import DocVec
    from docarray.array.any_array import AnyDocArray


class DocArraySummary:
    def __init__(self, docs: 'AnyDocArray'):
        self.docs = docs

    def summary(self) -> None:
        """
        Print a summary of this DocList object and a summary of the schema of its
        Document type.
        """
        from rich import box
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        from docarray.array import DocVec

        table = Table(box=box.SIMPLE, highlight=True)
        table.show_header = False
        table.add_row('Type', self.docs.__class__.__name__)
        table.add_row('Length', str(len(self.docs)), end_section=True)

        if isinstance(self.docs, DocVec):
            table.add_row('Stacked columns:')
            stacked_fields = self._get_stacked_fields(docs=self.docs)
            for field_name in stacked_fields:
                val = self.docs
                for attr in field_name.split('.'):
                    val = getattr(val, attr)

                if isinstance(val, AbstractTensor):
                    comp_be = val.get_comp_backend()
                    if comp_be.to_numpy(comp_be.isnan(val)).all():
                        col_2 = f'None ({val.__class__.__name__})'
                    else:
                        col_2 = (
                            f'{val.__class__.__name__} of shape {comp_be.shape(val)}'
                            f', dtype: {comp_be.dtype(val)}'
                        )
                        if comp_be.device(val):
                            col_2 += f', device: {comp_be.device(val)}'

                    table.add_row(f'  • {field_name}:', col_2)

        Console().print(Panel(table, title='DocList Summary', expand=False))
        self.docs.doc_type.schema_summary()

    @staticmethod
    def _get_stacked_fields(docs: 'DocVec') -> List[str]:  # TODO this might
        # broken
        """
        Return a list of the field names of a DocVec instance that are
        doc_vec, i.e. all the fields that are of type AbstractTensor. Nested field
        paths are separated by dot, such as: 'attr.nested_attr'.
        """
        fields = []
        for field_name, value_tens in docs._storage.tensor_columns.items():
            fields.append(field_name)
        for field_name, value_doc in docs._storage.doc_columns.items():
            if value_doc is not None:
                fields.extend(
                    [
                        f'{field_name}.{x}'
                        for x in DocArraySummary._get_stacked_fields(docs=value_doc)
                    ]
                )

        return fields
