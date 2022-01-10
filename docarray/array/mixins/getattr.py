from typing import List


class GetAttributeMixin:
    """Helpers that provide attributes getter in bulk """

    def _get_attributes(self, *fields: str) -> List:
        """Return all nonempty values of the fields from all docs this array contains

        :param fields: Variable length argument with the name of the fields to extract
        :return: Returns a list of the values for these fields.
            When `fields` has multiple values, then it returns a list of list.
        """
        e_index, b_index = None, None
        fields = list(fields)
        if 'embedding' in fields:
            e_index = fields.index('embedding')
        if 'blob' in fields:
            b_index = fields.index('blob')
            fields.remove('blob')

        if 'embedding' in fields:
            fields.remove('embedding')
        if 'blob' in fields:
            fields.remove('blob')

        if fields:
            contents = [doc._get_attributes(*fields) for doc in self]
            if len(fields) > 1:
                contents = list(map(list, zip(*contents)))
            if b_index is None and e_index is None:
                return contents

            if len(fields) == 1:
                contents = [contents]
            if b_index is not None:
                contents.insert(b_index, self.blobs)
            if e_index is not None:
                contents.insert(e_index, self.embeddings)
            return contents

        if b_index is not None and e_index is None:
            return self.blobs
        if b_index is None and e_index is not None:
            return self.embeddings
        if b_index is not None and e_index is not None:
            return (
                [self.embeddings, self.blobs]
                if b_index > e_index
                else [self.blobs, self.embeddings]
            )
