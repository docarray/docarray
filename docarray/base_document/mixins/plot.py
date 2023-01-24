from docarray.base_document.abstract_document import AbstractDocument
from docarray.plotting.document_summary import DocumentSummary


class PlotMixin(AbstractDocument):
    def summary(self) -> None:
        """Print non-empty fields and nested structure of this Document object."""
        DocumentSummary(doc=self).summary()

    @classmethod
    def schema_summary(cls) -> None:
        """Print a summary of the Documents schema."""
        DocumentSummary().schema_summary(cls)

    def _ipython_display_(self):
        """Displays the object in IPython as a side effect"""
        self.summary()
