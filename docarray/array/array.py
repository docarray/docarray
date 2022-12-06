from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
from typing import DefaultDict, Dict, Iterable, List, Optional, Type, TypeVar, Union

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

    wrapper.__doc__ = (
        wrapper.__doc__
        + ' \n This method is not available in stacked mode call {meth}.unstack() '
        + 'to use it'
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
    A DocumentArray is a container of Documents.

    :param docs: iterable of Document

    A DocumentArray is a list of Documents of any schema. However, many 
    DocumentArray features are only available if these Documents are
    homogeneous and follow the same schema. To be precise, in this schema you can use the
    `DocumentArray[MyDocument]` syntax where MyDocument is a Document class
    (i.e. schema). This creates a DocumentArray that can only contain Documents of
    the type Document.

    EXAMPLE USAGE
    .. code-block:: python
        from docarray import Document, DocumentArray
        from docarray.typing import NdArray, ImageUrl


        class Image(Document):
            tensor: Optional[NdArray[100]]
            url: ImageUrl


        da = DocumentArray[Image](Image(url='http://url.com/foo.png') for _ in range(10))


    If your DocumentArray is homogeneous (i.e. follows the same schema), you can access the
    field at the DocumentArray level (for example `da.tensor`). You can also set them,
    with `da.tensor = np.random.random([10, 100])`


    A DocumentArray can be in one of two modes: unstacked mode and stacked mode.


    **Unstacked mode (default)**:
    In this case a DocumentArray is a list of Documents and each Document owns its data.
    The getter and setter shown above returns a list of the fields of each Document
    (or a DocumentArray if the field is a nested Document). This list/DocumentArray is created on the fly. The setter sets the field of each Document to the value
    of the list/DocumentArray/Tensor passed as parameters.

    This list-like behavior is not always optimal, especially when you want
    to process data in batches or perform operations involving matrix computation.
    This is where the stack mode of the DocumentArray comes in handy.

    In **stacked mode** tensor-like fields of every Document are stored on the
    DocumentArray level, as one stacked tensor. This enables operations on the entire
    batch without iterating over the DocumentArray.
    In this mode the Documents in in the Document don't own the data anymore but just
    reference the data in the DocumentArray's tensor.
    Operations like `da.append` are not supported because
    they are too slow. For these operations you should use unstacked
    mode instead.

    To switch from stacked to unstacked mode (or vice-versa), call `da.unstack()` or
    `da.stack`. There are also two context manager for these modes:
    `with da.stack_mode():` and `with da.unstack_mode():`

    see {meth}`.stack` and {meth}`.unstack` for more information.

    You should use unstacked mode, if:

    * You want to insert, append, delete, or shuffle Documents in your DocumentArray
    * Intend to separate the DocumentArray into smaller batches later on

    You should use stacked mode, if you:

    * You want to process the entire DocumentArray as one batch
    * Are using the DocumentArray in an ML model for training or inference

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

    def stack(self: T) -> T:
        """
        Puts the DocumentArray into stacked mode.
        :return: itself

        When entering stacked mode the DocumentArray creates a column for each field
        of the Document that are Tensors or nested Documents that contain at
        least one Tensor field. This is useful when to perform operations on
        the whole array at once. In stacked mode, accessing or setting the DocumentArray's fields
        DocumentArray accesses or sets the column of the array.

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

            columns_to_stack: DefaultDict[
                str, Union[List[TorchTensor], List[NdArray], List[BaseDocument]]
            ] = defaultdict(
                list
            )  # type: ignore

            for doc in self:
                for field_to_stack in self._columns.keys():
                    columns_to_stack[field_to_stack].append(
                        getattr(doc, field_to_stack)
                    )
                    setattr(doc, field_to_stack, None)

            for field_to_stack, to_stack in columns_to_stack.items():

                type_ = self.document_type.__fields__[field_to_stack].type_
                if issubclass(type_, BaseDocument):
                    self._columns[field_to_stack] = DocumentArray[type_](  # type: ignore # noqa: E501
                        to_stack
                    ).stack()
                else:
                    self._columns[field_to_stack] = type_.__docarray_stack__(to_stack)

            for i, doc in enumerate(self):
                for field_to_stack in self._columns.keys():
                    type_ = self.document_type.__fields__[field_to_stack].type_
                    if issubclass(type_, TorchTensor) or issubclass(type_, NdArray):
                        if self._columns is not None:
                            setattr(doc, field_to_stack, self._columns[field_to_stack])
                        else:
                            raise ValueError(
                                f'The DocumentArray is in stacked mode, '
                                f'but no stacked data is '
                                f'present for {field_to_stack}. This is inconsistent'
                            )

        return self

    def unstack(self: T) -> T:
        """
        Puts the DocumentArray into unstacked mode.
        :return: itself


        Calling unstack will unstack all the columns of the DocumentArray and restore the
        data of each Document in the DocumentArray.

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
            # this iterates over the ndarray and assigns each row to a Document

            print(batch[0].tensor[0])
            # >>> 1

        see {meth}`.stack` for more information on switching to stack mode
        """
        if self.is_stacked() and self._columns:
            for field in list(self._columns.keys()):
                # list needed here otherwise we are modifying the dict while iterating
                del self._columns[field]

            self._columns = None

        return self

    @contextmanager
    def stacked_mode(self):
        """
        Context manager to put the DocumentArray in stack mode and unstack it when
        exiting the context manager.

        EXAMPLE USAGE
        .. code-block:: python
            with da.stacked_mode():
                ...
        """
        try:
            yield self.stack()
        finally:
            self.unstack()

    @contextmanager
    def unstacked_mode(self):
        """
        Context manager to put the DocumentArray in unstacked mode and stack it when
        exiting the context manager.

        EXAMPLE USAGE
        .. code-block:: python
            with da.unstacked_mode():
                ...
        """
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
        return the list of fields that are columns of the DocumentArray
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
