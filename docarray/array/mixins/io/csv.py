import csv
from contextlib import nullcontext
from typing import Union, TextIO, Optional, Dict, TYPE_CHECKING, Type, Sequence

import numpy as np

if TYPE_CHECKING:
    from ....typing import T


class CsvIOMixin:
    """CSV IO helper.

    can be applied to DA & DAM
    """

    def save_embeddings_csv(
        self, file: Union[str, TextIO], encoding: str = 'utf-8', **kwargs
    ) -> None:
        """Save embeddings to a CSV file

        This function utilizes :meth:`numpy.savetxt` internal.

        :param file: File or filename to which the data is saved.
        :param encoding: encoding used to save the data into a file. By default, ``utf-8`` is used.
        :param kwargs: extra kwargs will be passed to :meth:`numpy.savetxt`.
        """
        if hasattr(file, 'write'):
            file_ctx = nullcontext(file)
        else:
            file_ctx = open(file, 'w', encoding=encoding)
        with file_ctx:
            np.savetxt(file_ctx, self.embeddings, **kwargs)

    def save_csv(
        self,
        file: Union[str, TextIO],
        flatten_tags: bool = True,
        exclude_fields: Optional[Sequence[str]] = None,
        dialect: Union[str, 'csv.Dialect'] = 'excel',
        with_header: bool = True,
        encoding: str = 'utf-8',
    ) -> None:
        """Save array elements into a CSV file.

        :param file: File or filename to which the data is saved.
        :param flatten_tags: if set, then all fields in ``Document.tags`` will be flattened into ``tag__fieldname`` and
            stored as separated columns. It is useful when ``tags`` contain a lot of information.
        :param exclude_fields: if set, those fields wont show up in the output CSV
        :param dialect: define a set of parameters specific to a particular CSV dialect. could be a string that represents
            predefined dialects in your system, or could be a :class:`csv.Dialect` class that groups specific formatting
            parameters together.
        :param encoding: encoding used to save the data into a CSV file. By default, ``utf-8`` is used.
        """
        if hasattr(file, 'write'):
            file_ctx = nullcontext(file)
        else:
            file_ctx = open(file, 'w', encoding=encoding)

        with file_ctx as fp:
            if flatten_tags and self[0].tags:
                keys = list(self[0].non_empty_fields) + list(
                    f'tag__{k}' for k in self[0].tags
                )
                keys.remove('tags')
            else:
                flatten_tags = False
                keys = list(self[0].non_empty_fields)

            if exclude_fields:
                for k in exclude_fields:
                    if k in keys:
                        keys.remove(k)

            writer = csv.DictWriter(fp, fieldnames=keys, dialect=dialect)

            if with_header:
                writer.writeheader()

            for d in self:
                pd = d.to_dict(
                    protocol='jsonschema',
                    exclude=set(exclude_fields) if exclude_fields else None,
                    exclude_none=True,
                )
                if flatten_tags:
                    t = pd.pop('tags')
                    pd.update({f'tag__{k}': v for k, v in t.items()})
                writer.writerow(pd)

    @classmethod
    def load_csv(
        cls: Type['T'],
        file: Union[str, TextIO],
        field_resolver: Optional[Dict[str, str]] = None,
        encoding: str = 'utf-8',
    ) -> 'T':
        """Load array elements from a binary file.

        :param file: File or filename to which the data is saved.
        :param field_resolver: a map from field names defined in JSON, dict to the field
            names defined in Document.
        :param encoding: encoding used to read a CSV file. By default, ``utf-8`` is used.
        :return: a DocumentArray object
        """

        from ....document.generators import from_csv

        return cls(from_csv(file, field_resolver=field_resolver, encoding=encoding))
