from typing import Iterator, Dict


class Offset2ID:
    def __init__(self):
        self._ids = []

    def get_id(self, idx):
        return self._ids[idx]

    def append(self, data):
        self._ids.append(data)

    def extend(self, data):
        self._ids.extend(data)

    def update(self, position, data_id):
        self._ids[position] = data_id

    def delete_by_id(self, _id):
        del self._ids[self._ids.index(_id)]

    def index(self, _id):
        return self._ids.index(_id)

    def delete_by_offset(self, position):
        del self._ids[position]

    def insert(self, position, data_id):
        self._ids.insert(position, data_id)

    def clear(self):
        self._ids.clear()

    def delete_by_ids(self, ids):
        ids = set(ids)
        self._ids = list(filter(lambda _id: _id not in ids, self._ids))

    def update_ids(self, _ids_map: Dict[str, str]):
        for i in range(len(self._ids)):
            if self._ids[i] in _ids_map:
                self._ids[i] = _ids_map[self._ids[i]]

    def save(self):
        pass

    def load(self):
        pass

    def __iter__(self) -> Iterator['str']:
        yield from self._ids

    def __eq__(self, other):
        return self._ids == other._ids

    def __len__(self):
        return len(self._ids)
