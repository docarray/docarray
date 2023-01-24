from typing import Optional

import torch

from docarray import BaseDocument, DocumentArray
from docarray.data import TorchDataset
from docarray.documents import Image, Text


class ImagePreprocess:
    def __call__(self, image: Image) -> None:
        assert isinstance(image, Image)
        image.tensor = torch.randn(3, 32, 32)


class TextPreprocess:
    def __call__(self, text: Text) -> None:
        assert isinstance(text, Text)
        text.embedding = torch.randn(64)


def test_torch_dataset():
    BATCH_SIZE = 32

    class PairTextImage(BaseDocument):
        text: Text
        image: Image

    class Mydataset(TorchDataset[PairTextImage]):
        def __init__(self, csv_file: str, preprocessing: Optional[dict] = None):
            with open(csv_file, "r") as f:
                f.readline()
                da = DocumentArray[PairTextImage](
                    PairTextImage(
                        text=Text(text=i[1]),
                        image=Image(url=f"tests/toydata/image-data/{i[0]}"),
                    )
                    for i in map(lambda x: x.strip().split(","), f.readlines())
                )
            super().__init__(da, preprocessing)

    from torch.utils.data import DataLoader

    preprocessing = {"image": ImagePreprocess(), "text": TextPreprocess()}
    dataset = Mydataset(
        csv_file="tests/toydata/captions.csv", preprocessing=preprocessing
    )
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, collate_fn=dataset.collate_fn, shuffle=True
    )

    from docarray.array.array_stacked import DocumentArrayStacked

    # TEMPORARY PATCH WHILE WE WAIT FOR ISINSTANCE TO BE FIXED
    isinstance = lambda x, y: str(x.__class__) == str(y)  # noqa: E731

    for batch in loader:
        assert isinstance(batch, DocumentArrayStacked[PairTextImage])
        assert len(batch) == BATCH_SIZE
