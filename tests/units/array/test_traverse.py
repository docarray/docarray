from typing import Optional

import pytest
import torch

from docarray import BaseDocument, DocumentArray
from docarray.array.abstract_array import AnyDocumentArray
from docarray.documents import TextDoc
from docarray.typing import TorchTensor

num_docs = 5
num_sub_docs = 2
num_sub_sub_docs = 3


@pytest.fixture
def multi_model_docs():
    class SubSubDoc(BaseDocument):
        sub_sub_text: TextDoc
        sub_sub_tensor: TorchTensor[2]

    class SubDoc(BaseDocument):
        sub_text: TextDoc
        sub_da: DocumentArray[SubSubDoc]

    class MultiModalDoc(BaseDocument):
        mm_text: TextDoc
        mm_tensor: Optional[TorchTensor[3, 2, 2]]
        mm_da: DocumentArray[SubDoc]

    docs = DocumentArray[MultiModalDoc](
        [
            MultiModalDoc(
                mm_text=TextDoc(text=f'hello{i}'),
                mm_da=[
                    SubDoc(
                        sub_text=TextDoc(text=f'sub_{i}_1'),
                        sub_da=DocumentArray[SubSubDoc](
                            [
                                SubSubDoc(
                                    sub_sub_text=TextDoc(text='subsub'),
                                    sub_sub_tensor=torch.zeros(2),
                                )
                                for _ in range(num_sub_sub_docs)
                            ]
                        ),
                    )
                    for _ in range(num_sub_docs)
                ],
            )
            for i in range(num_docs)
        ]
    )

    return docs


@pytest.mark.parametrize(
    'access_path,len_result',
    [
        ('mm_text', num_docs),  # List of 5 Text objs
        ('mm_text__text', num_docs),  # List of 5 strings
        ('mm_da', num_docs * num_sub_docs),  # List of 5 * 2 SubDoc objs
        ('mm_da__sub_text', num_docs * num_sub_docs),  # List of 5 * 2 Text objs
        (
            'mm_da__sub_da',
            num_docs * num_sub_docs * num_sub_sub_docs,
        ),  # List of 5 * 2 * 3 SubSubDoc objs
        (
            'mm_da__sub_da__sub_sub_text',
            num_docs * num_sub_docs * num_sub_sub_docs,
        ),  # List of 5 * 2 * 3 Text objs
    ],
)
def test_traverse_flat(multi_model_docs, access_path, len_result):
    traversed = multi_model_docs.traverse_flat(access_path)
    assert len(traversed) == len_result


def test_traverse_stacked_da():
    class Image(BaseDocument):
        tensor: TorchTensor[3, 224, 224]

    batch = DocumentArray[Image](
        [
            Image(
                tensor=torch.zeros(3, 224, 224),
            )
            for _ in range(2)
        ]
    )

    batch_stacked = batch.stack()
    tensors = batch_stacked.traverse_flat(access_path='tensor')

    assert tensors.shape == (2, 3, 224, 224)
    assert isinstance(tensors, torch.Tensor)


@pytest.mark.parametrize(
    'input_list,output_list',
    [
        ([1, 2, 3], [1, 2, 3]),
        ([[1], [2], [3]], [1, 2, 3]),
        ([[[1]], [[2]], [[3]]], [[1], [2], [3]]),
    ],
)
def test_flatten_one_level(input_list, output_list):
    flattened = AnyDocumentArray._flatten_one_level(sequence=input_list)
    assert flattened == output_list


def test_flatten_one_level_list_of_da():
    doc = BaseDocument()
    input_list = [DocumentArray([doc, doc, doc])]

    flattened = AnyDocumentArray._flatten_one_level(sequence=input_list)
    assert flattened == [doc, doc, doc]
