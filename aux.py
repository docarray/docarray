from typing import Sequence, TypeVar, Any, Callable, get_args, Generic

from pydantic_core import core_schema, ValidationError

from pydantic import BaseModel

T = TypeVar('T')


class MySequence(Sequence[T], Generic[T]):
    def __init__(self, v: Sequence[T]):
        self.v = v

    def __getitem__(self, i):
        return self.v[i]

    def __len__(self):
        return len(self.v)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: Callable[[Any], core_schema.CoreSchema]
    ) -> core_schema.CoreSchema:
        print(f'source {source} and MySeq handler {handler}')
        instance_schema = core_schema.is_instance_schema(cls)

        args = get_args(source)
        print(f'args1 {args}')
        if args:
            sequence_t_schema = handler(Sequence[args[0]])
        else:
            sequence_t_schema = handler(Sequence)

        non_instance_schema = core_schema.with_info_after_validator_function(
            lambda v, i: MySequence(v), sequence_t_schema
        )
        return core_schema.union_schema([instance_schema, non_instance_schema])


class MySequence2(MySequence, Generic[T]):
    pass


class A(BaseModel):
    b: int

class M(BaseModel):
    model_config = dict(validate_default=True)

    s1: MySequence2[A]


print(M.schema())

args = get_args(MySequence2[A])
print(f'MySequence2 args {args}')

from typing import List, Union
from docarray.array.any_array import AnyDocArray
from docarray import BaseDoc, DocList
import pydantic


class Doc(BaseDoc):
    a: str



print(f'Doc {Doc.schema()}')


class DocDoc(BaseDoc):
    docs: DocList[Doc]


print(DocDoc.schema())

args = get_args(DocList[Doc])
print(f'DocList args {args}')


args = get_args(AnyDocArray[Doc])
print(f'AnyDocArray args {args}')













