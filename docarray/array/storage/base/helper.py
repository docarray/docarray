from typing import Iterator, List, Tuple, Dict


class Offset2ID:
    def __init__(self):
        self.offset2id = []

    def get_id(self, idx):
        return self.offset2id[idx]

    def append(self, data):
        self.offset2id.append(data)

    def extend(self, data):
        self.offset2id.extend(data)

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

    def clear(self):
        self.offset2id.clear()

    def delete_by_ids(self, ids):
        ids = set(ids)
        self.offset2id = list(filter(lambda _id: _id not in ids, self.offset2id))

    def update_ids(self, _ids_map: Dict[str, str]):
        for i in range(len(self.offset2id)):
            if self.offset2id[i] in _ids_map:
                self.offset2id[i] = _ids_map[self.offset2id[i]]

    def save(self):
        pass

    def load(self):
        pass

    def __iter__(self) -> Iterator['str']:
        yield from self.offset2id

    def __eq__(self, other):
        return self.offset2id == other.offset2id

    def __len__(self):
        return len(self.offset2id)
