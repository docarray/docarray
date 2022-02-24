from typing import Iterator, Dict


class Offset2ID:
    def __init__(self, ids=None):
        self.ids = ids or []

    def get_id(self, idx):
        return self.ids[idx]

    def append(self, data):
        self.ids.append(data)

    def extend(self, data):
        self.ids.extend(data)

    def update(self, position, data_id):
        self.ids[position] = data_id

    def delete_by_id(self, _id):
        del self.ids[self.ids.index(_id)]

    def index(self, _id):
        return self.ids.index(_id)

    def delete_by_offset(self, position):
        del self.ids[position]

    def insert(self, position, data_id):
        self.ids.insert(position, data_id)

    def clear(self):
        self.ids.clear()

    def delete_by_ids(self, ids):
        ids = set(ids)
        self.ids = list(filter(lambda _id: _id not in ids, self.ids))

    def update_ids(self, _ids_map: Dict[str, str]):
        for i in range(len(self.ids)):
            if self.ids[i] in _ids_map:
                self.ids[i] = _ids_map[self.ids[i]]

    def save(self):
        pass

    def load(self):
        pass

    def __iter__(self) -> Iterator['str']:
        yield from self.ids

    def __eq__(self, other):
        return self.ids == other.ids

    def __len__(self):
        return len(self.ids)
