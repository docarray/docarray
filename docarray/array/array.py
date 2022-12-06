from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
from typing import Dict, Iterable, List, Optional, Type, TypeVar, Union

from docarray.array.abstract_array import AbstractDocumentArray
from docarray.array.mixins import GetAttributeArrayMixin, ProtoArrayMixin
from docarray.document import AnyDocument, BaseDocument, BaseNode
from docarray.typing import NdArray, TorchTensor


def _stacked_mode_blocker(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.is_stacked():
            raise RuntimeError(
                f'Cannot call {func.__name__} when the document array is in stack mode'
            )
        return func(self, *args, **kwargs)

    (
        wrapper.__doc__
        + ' \n This method is not available in stacked mode call {meth}.unstack() '
        'to use it'
    )
    return wrapper


T = TypeVar('T', bound='DocumentArray')


class DocumentArray(
    list,
    ProtoArrayMixin,
    GetAttributeArrayMixin,
    AbstractDocumentArray,
    BaseNode,
):
    """
    a DocumentArray is a container of Document of the same schema.

    :param docs: iterable of Document


    A DocumentArray can only contain Documents that follow the same schema. To precise
    this schema you can use the `DocumentArray[Document]` syntax. This will create a
    DocumentArray that can only contain Document of the type Document. (Note that there
    exists a special schema (AnySchema) that allows any Document to be stored in the
    DocumentArray, but this is not recommended to use).

    EXAMPLE USAGE
    .. code-block:: python
        from docarray import Document, DocumentArray
        from docarray.typing import NdArray, ImageUrl


        class Image(Document):
            tensor: Optional[NdArray[100]]
            url: ImageUrl


        da = DocumentArray[Image](Image(url='http://url.com') for _ in range(10))


    DocumentArray defines setter and getter for each field of the Document schema. These
    getters and setters are defined dynamically at runtime. This allows to access the
    field of the Document in a natural way. For example, if you have a DocumentArray of
    Image you can do: `da.tensor` to get the tensor of all the Image in the
    DocumentArray. You can also do `da.tensor = np.random.random([10, 100])` to set the
    tensor of all the Image.


    A DocumentArray can be in one of two modes: unstacked mode and stacked mode.


    **Unstacked mode (default)**:
    In this case a DocumentArray is a list of Document and each Document owns its data.
    The getter and setter shown above will return a list of the fields of each Document
    (or a DocumentArray if the field is a nested Document). This list/DocumentArray will
    be created on the fly. The setter will set the field of each Document to the value
    of the list/DocumentArray/Tensor passed as parameters.

    Nevertheless, this list-like behavior is not always optimal especially when you want
    to process you data in batch and do operation which involves matrix computation,This
    is where the stack mode of the DocumentArray comes in handy.

    In **stacked mode** each field which are Tensor are stored as a column in a tensor
    of batch the size of the DocumentArray. This allows to do operation on the whole
    batch instead of iterating over the DocumentArray.
    In this mode the Document inside in the Document don't own the data anymore but just
    reference to the data in the tensor of the DocumentArray.
    In stacked mode the getter and setter just replace the tensor of the DocumentArray
    of the given field.
    Operation like `da.append` are not allowed anymore because
    they are too slow and not recommended to use. You should rather use the unstacked
    mode.

    To switch from stacked mode to unstacked mode you need to call `da.unstack()` and
    `da.stack`. There are as well two context manager to for these modes.
    `with da.stack_mode():` and `with da.unstack_mode():`

    see {meth}`.stack` and {meth}`.unstack` for more information.
    """

    document_type: Type[BaseDocument] = AnyDocument

    def __init__(self, docs: Iterable[BaseDocument]):
        super().__init__(doc_ for doc_ in docs)

        self._columns: Optional[
            Dict[str, Union[TorchTensor, AbstractDocumentArray, NdArray, None]]
        ] = None

    def __class_getitem__(cls, item: Type[BaseDocument]):
        if not issubclass(item, BaseDocument):
            raise ValueError(
                f'DocumentArray[item] item should be a Document not a {item} '
            )

        class _DocumentArrayTyped(cls):  # type: ignore
            document_type: Type[BaseDocument] = item

        for field in _DocumentArrayTyped.document_type.__fields__.keys():

            def _property_generator(val: str):
                def _getter(self):
                    return self._get_array_attribute(val)

                def _setter(self, value):
                    self._set_array_attribute(val, value)

                # need docstring for the property
                return property(fget=_getter, fset=_setter)

            setattr(_DocumentArrayTyped, field, _property_generator(field))
            # this generates property on the fly based on the schema of the item

        _DocumentArrayTyped.__name__ = f'DocumentArray[{item.__name__}]'
        _DocumentArrayTyped.__qualname__ = f'DocumentArray[{item.__name__}]'

        return _DocumentArrayTyped

    def __getitem__(self, item):
        if self.is_stacked():
            doc = super().__getitem__(item)
            # NOTE: this could be speed up by using a cache
            for field in self._columns.keys():
                setattr(doc, field, self._columns[field][item])
            return doc
        else:
            return super().__getitem__(item)

    def stack(self):
        """
        Calling this method will make the DocumentArray enter stacked mode. You should
        call this method only if the DocumentArray is already in unstacked mode
        (the default mode). (Calling it while being already in stacked mode will have
        no effect)

        When entering stack mode the DocumentArray will create a column for each field
        of the Document that are Tensor or that are Nested Document that contains at
        least one Tensor field. This is useful when you want to perform operations on
        the whole array at once. In stack mode, accessing or setting fields of the
        DocumentArray will access or set the column of the array.


        IMPORTANT: in stacked_mode you cannot add or remove Document from the array.
        You can only modify the fields of the DocumentArray. This is intentionally done
        because extending or inserting a column is slow. You should rather use the
        unstack mode for such operation.

        EXAMPLE USAGE
        .. code-block:: python
            from docarray import Document, DocumentArray
            from docarray.typing import NdArray


            class Image(Document):
                tensor: NdArray[100]


            batch = DocumentArray[Image](
                [Image(tensor=np.zeros((100))) for _ in range(10)]
            )  # noqa: E510

            batch.stack()

            print(batch[0].tensor[0])
            # >>> 0

            print(batch.tensor.shape)
            # >>> (10, 3, 224, 224)

            batch.tensor = np.ones((10, 100))

            print(batch[0].tensor[0])
            # >>> 1

            batch.append(Image(tensor=np.zeros((100))))
            # >>> raise RuntimeError('Cannot call append when the document array is in
            # >>> stack mode'

        see {meth}`.unstack` for more information on how to switch to unstack mode
        """

        if not (self.is_stacked()):

            self._columns = dict()

            for field_name, field in self.document_type.__fields__.items():
                if (
                    issubclass(field.type_, TorchTensor)
                    or issubclass(field.type_, BaseDocument)
                    or issubclass(field.type_, NdArray)
                ):
                    self._columns[field_name] = None

            columns_to_stack: Dict[
                str, Union[List[TorchTensor], List[NdArray], List[BaseDocument]]
            ] = defaultdict(list)

            for doc in self:
                for field_to_stack in self._columns.keys():
                    columns_to_stack[field_to_stack].append(
                        getattr(doc, field_to_stack)
                    )
                    setattr(doc, field_to_stack, None)

            for field_to_stack, to_stack in columns_to_stack.items():

                type_ = self.document_type.__fields__[field_to_stack].type_
                if issubclass(type_, BaseDocument):
                    self._columns[field_to_stack] = DocumentArray[type_](
                        to_stack
                    ).stack()
                else:
                    self._columns[field_to_stack] = type_.__docarray_stack__(to_stack)

            for i, doc in enumerate(self):
                for field_to_stack in self._columns.keys():
                    type_ = self.document_type.__fields__[field_to_stack].type_
                    if issubclass(type_, TorchTensor) or issubclass(type_, NdArray):
                        setattr(doc, field_to_stack, self._columns[field_to_stack][i])

        return self

    def unstack(self):
        """
        Calling this method will make the DocumentArray enter in unstack mode.
        This is the default mode of any DocumentArray. This means you should call this
        only method if the DocumentArray is already in stacked mode. (Calling it while
        being in unstacked mode will have no effect)

        Calling unstack will unstack all the columns of the DocumentArray and put the
        data back in each Document of the DocumentArray.

        EXAMPLE USAGE
        .. code-block:: python
            from docarray import Document, DocumentArray, Text
            from docarray.typing import NdArray


            class Image(Document):
                tensor: NdArray[100]


            batch = DocumentArray[Image](
                [Image(tensor=np.zeros((100))) for _ in range(10)]
            )  # noqa: E510

            batch.stack()
            batch.unstack()
            print(type(batch.tensor))
            # >>> list
            print(len(batch.tensor))
            # >>> 10

            batch.append(Image(tensor=np.zeros((100))))
            # this work just fine

            batch.tensor = np.ones((10, 100))
            # this iterate over the ndarray and assign the row to each Document

            print(batch[0].tensor[0])
            # >>> 1

        see {meth}`.stack` for more information on how to switch to stack mode
        """
        if self.is_stacked():

            for field in list(self._columns.keys()):
                # list needed here otherwise we are modifying the dict while iterating
                del self._columns[field]

            self._columns = None

        return self

    @contextmanager
    def stacked_mode(self):
        try:
            yield self.stack()
        finally:
            self.unstack()

    @contextmanager
    def unstacked_mode(self):
        try:
            yield self.unstack()
        finally:
            self.stack()

    def is_stacked(self) -> bool:
        """
        Return True if the document array is in stack mode
        """
        return self._columns is not None

    def _column_fields(self) -> List[str]:
        """
        return the field store as columns
        :return: the list of keys of the columns
        """
        if self.is_stacked() and self._columns is not None:
            # need to repeat myself here bc of mypy
            return list(self._columns.keys())
        else:
            return []

    append = _stacked_mode_blocker(list.append)
    extend = _stacked_mode_blocker(list.extend)
    clear = _stacked_mode_blocker(list.clear)
    insert = _stacked_mode_blocker(list.insert)
    pop = _stacked_mode_blocker(list.pop)
    remove = _stacked_mode_blocker(list.remove)
    reverse = _stacked_mode_blocker(list.reverse)
    sort = _stacked_mode_blocker(list.sort)
