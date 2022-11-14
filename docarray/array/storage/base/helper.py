from typing import Iterator, Dict


class Offset2ID:
    def __init__(self, ids=None, list_like=True):
        self.ids = ids or []
        self._list_like = list_like

    def get_id(self, idx):
        if not self._list_like:
            raise ValueError(
                "The offset2id is not enabled for list-like indexes. To avoid this error, configure the "
                "`list_like` to True"
            )
        return self.ids[idx]

    def append(self, data):
        if self._list_like:
            self.ids.append(data)

    def extend(self, data):
        if self._list_like:
            self.ids.extend(data)

    def update(self, position, data_id):
        if self._list_like:
            self.ids[position] = data_id

    def delete_by_id(self, _id):
        if self._list_like:
            del self.ids[self.ids.index(_id)]

    def index(self, _id):
        if not self._list_like:
            raise ValueError(
                "The offset2id is not enabled for list-like indexes. To avoid this error, configure the "
                "`list_like` to True"
            )
        return self.ids.index(_id)

    def delete_by_offset(self, position):
        if self._list_like:
            del self.ids[position]

    def insert(self, position, data_id):
        if self._list_like:
            self.ids.insert(position, data_id)

    def clear(self):
        if self._list_like:
            self.ids.clear()

    def delete_by_ids(self, ids):
        if self._list_like:
            ids = set(ids)
            self.ids = list(filter(lambda _id: _id not in ids, self.ids))

    def update_ids(self, _ids_map: Dict[str, str]):
        if self._list_like:
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
