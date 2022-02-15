from typing import Iterator


class Offset2ID:
    def __init__(self):
        self.offset2id = []

    def get_id(self, idx):
        return self.offset2id[idx]

    def append(self, data):
        self.offset2id.append(data)

    def update(self, position, data_id):
        self.offset2id[position] = data_id

    def delete_by_id(self, _id):
        del self.offset2id[self.offset2id.index(_id)]

    def index(self, _id):
        return self.offset2id.index(_id)

    def delete_by_offset(self, position):
        del self.offset2id[position]

    def insert(self, position, data_id):
        self.offset2id.insert(position, data_id)

    def save(self):
        pass

    def load(self):
        pass

    def __iter__(self) -> Iterator['str']:
        yield from self.offset2id
