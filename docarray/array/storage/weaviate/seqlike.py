from typing import Union, Iterable

from ..base.seqlike import BaseSequenceLikeMixin
from ..registry import _REGISTRY
from .... import Document


class SequenceLikeMixin(BaseSequenceLikeMixin):
    """Implement sequence-like methods for DocumentArray with weaviate as storage"""

    def __eq__(self, other):
        """Compare this object to the other, returns True if and only if other
        as the same type as self and other has the same meta information

        :param other: the other object to check for equality
        :return: ``True`` if other is equal to self
        """
        # two DAW are considered as the same if they have the same client meta data
        return (
            type(self) is type(other)
            and self._client.get_meta() == other._client.get_meta()
            and self._config == other._config
        )

    def __len__(self):
        """Return the length of :class:`DocumentArray` that uses weaviate as storage

        :return: the length of this :class:`DocumentArrayWeaviate` object
        """
        cls_data = (
            self._client.query.aggregate(self._class_name)
            .with_meta_count()
            .do()
            .get('data', {})
            .get('Aggregate', {})
            .get(self._class_name, [])
        )

        if not cls_data:
            return 0

        return cls_data[0]['meta']['count']

    def __contains__(self, x: Union[str, 'Document']):
        """Check if ``x`` is contained in this :class:`DocumentArray` with weaviate storage

        :param x: the id of the document to check or the document object itself
        :return: True if ``x`` is contained in self
        """
        if isinstance(x, str):
            return self._client.data_object.exists(self._map_id(x))
        elif isinstance(x, Document):
            return self._client.data_object.exists(self._map_id(x.id))
        else:
            return False

    def __del__(self):
        """Delete this :class:`DocumentArrayWeaviate` object"""
        super().__del__()
        if (
            not self._persist
            and len(_REGISTRY[self.__class__.__name__][self._class_name]) == 1
        ):
            self._client.schema.delete_class(self._class_name)
            self._client.schema.delete_class(self._meta_name)
        _REGISTRY[self.__class__.__name__][self._class_name].remove(self)

    def __repr__(self):
        """Return the string representation of :class:`DocumentArrayWeaviate` object
        :return: string representation of this object
        """
        return f'<{self.__class__.__name__} (length={len(self)}) at {id(self)}>'

    def extend(self, values: Iterable['Document']) -> None:
        """Extends the array with the given values

        :param values: Documents to be added
        """
        with self._client.batch(batch_size=50) as _b:
            for d in values:
                _b.add_data_object(**self._doc2weaviate_create_payload(d))
                self._offset2ids.append(d.id)
