import pytest
import torch
from torch.utils.data import DataLoader

from docarray import BaseDocument, DocumentArray
from docarray.data import MultiModalDataset
from docarray.documents import ImageDoc, Text


class PairTextImage(BaseDocument):
    text: Text
    image: ImageDoc


class ImagePreprocess:
    def __call__(self, image: ImageDoc) -> None:
        assert isinstance(image, ImageDoc)
        image.tensor = torch.randn(3, 32, 32)


class TextPreprocess:
    def __call__(self, text: Text) -> None:
        assert isinstance(text, Text)
        if text.text.endswith(' meow'):
            text.embedding = torch.randn(42)
        else:
            text.embedding = torch.randn(64)


class Meowification:
    def __call__(self, text: str) -> None:
        assert isinstance(text, str)
        return text + ' meow'


@pytest.fixture
def captions_da() -> DocumentArray[PairTextImage]:
    with open("tests/toydata/captions.csv", "r") as f:
        f.readline()
        da = DocumentArray[PairTextImage](
            PairTextImage(
                text=Text(text=i[1]),
                image=ImageDoc(url=f"tests/toydata/image-data/{i[0]}"),
            )
            for i in map(lambda x: x.strip().split(","), f.readlines())
        )
    return da


def test_torch_dataset(captions_da: DocumentArray[PairTextImage]):
    BATCH_SIZE = 32

    preprocessing = {"image": ImagePreprocess(), "text": TextPreprocess()}
    dataset = MultiModalDataset[PairTextImage](captions_da, preprocessing)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, collate_fn=dataset.collate_fn, shuffle=True
    )

    from docarray.array.stacked.array_stacked import DocumentArrayStacked

    batch_lens = []
    for batch in loader:
        assert isinstance(batch, DocumentArrayStacked[PairTextImage])
        batch_lens.append(len(batch))
    assert all(x == BATCH_SIZE for x in batch_lens[:-1])


def test_primitives(captions_da: DocumentArray[PairTextImage]):
    BATCH_SIZE = 32

    preprocessing = {"text": Meowification()}
    dataset = MultiModalDataset[Text](captions_da.text, preprocessing)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, collate_fn=dataset.collate_fn, shuffle=True
    )

    batch = next(iter(loader))
    assert all(t.endswith(' meow') for t in batch.text)


def test_root_field(captions_da: DocumentArray[Text]):
    BATCH_SIZE = 32

    preprocessing = {"": TextPreprocess()}
    dataset = MultiModalDataset[Text](captions_da.text, preprocessing)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, collate_fn=dataset.collate_fn, shuffle=True
    )

    batch = next(iter(loader))
    assert batch.embedding.shape[1] == 64


def test_nested_field(captions_da: DocumentArray[PairTextImage]):
    BATCH_SIZE = 32

    preprocessing = {
        "image": ImagePreprocess(),
        "text": TextPreprocess(),
        "text.text": Meowification(),
    }
    dataset = MultiModalDataset[PairTextImage](captions_da, preprocessing)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, collate_fn=dataset.collate_fn, shuffle=True
    )

    batch = next(iter(loader))
    assert batch.text.embedding.shape[1] == 64

    preprocessing = {
        "image": ImagePreprocess(),
        "text.text": Meowification(),
        "text": TextPreprocess(),
    }
    dataset = MultiModalDataset[PairTextImage](captions_da, preprocessing)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, collate_fn=dataset.collate_fn, shuffle=True
    )

    batch = next(iter(loader))
    assert batch.text.embedding.shape[1] == 42


def test_torch_dl_multiprocessing(captions_da: DocumentArray[PairTextImage]):
    BATCH_SIZE = 32

    preprocessing = {"image": ImagePreprocess(), "text": TextPreprocess()}
    dataset = MultiModalDataset[PairTextImage](captions_da, preprocessing)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=dataset.collate_fn,
        shuffle=True,
        num_workers=2,
        multiprocessing_context='fork',
    )

    from docarray.array.stacked.array_stacked import DocumentArrayStacked

    batch_lens = []
    for batch in loader:
        assert isinstance(batch, DocumentArrayStacked[PairTextImage])
        batch_lens.append(len(batch))
    assert all(x == BATCH_SIZE for x in batch_lens[:-1])


@pytest.mark.skip(reason="UNRESOLVED BUG")
def test_torch_dl_pin_memory(captions_da: DocumentArray[PairTextImage]):
    BATCH_SIZE = 32

    preprocessing = {"image": ImagePreprocess(), "text": TextPreprocess()}
    dataset = MultiModalDataset[PairTextImage](captions_da, preprocessing)
    # Loader fails if dataset is empty
    if len(dataset) == 0:
        return
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=dataset.collate_fn,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
        multiprocessing_context='fork',
    )

    from docarray.array.stacked.array_stacked import DocumentArrayStacked

    batch_lens = []
    for batch in loader:
        assert isinstance(batch, DocumentArrayStacked[PairTextImage])
        batch_lens.append(len(batch))
    assert all(x == BATCH_SIZE for x in batch_lens[:-1])
