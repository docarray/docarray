from typing import Optional

import pandas as pd
import torchvision
from transformers import AutoTokenizer

from docarray import BaseDocument, DocumentArray
from docarray.data import TorchDataset
from docarray.documents import Image
from docarray.documents import Text
from docarray.documents import Text as BaseText
from docarray.typing import TorchTensor


class Tokens(BaseDocument):
    input_ids: TorchTensor[48]
    attention_mask: TorchTensor


class Text(BaseText):
    tokens: Optional[Tokens]


class ImagePreprocess:
    def __init__(self):
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(232),
                torchvision.transforms.RandomCrop(224),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, image: Image) -> None:
        image.tensor = self.transform(image.url.load())


class TextPreprocess:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")

    def __call__(self, text: Text) -> None:
        text.tokens = Tokens(
            **self.tokenizer(
                text.text, padding="max_length", truncation=True, max_length=48
            )
        )


def test_torch_dataset():
    BATCH_SIZE = 32

    class PairTextImage(BaseDocument):
        text: Text
        image: Image

    class Mydataset(TorchDataset[PairTextImage]):
        def __init__(self, csv_file: str, preprocessing: Optional[dict] = None):
            df = pd.read_csv(csv_file)
            da = DocumentArray[PairTextImage](
                PairTextImage(
                    text=Text(text=i.caption),
                    image=Image(url=f"tests/toydata/image-data/{i.image}"),
                )
                for i in df.itertuples()
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
