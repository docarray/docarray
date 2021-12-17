class RepresentMixin:
    def __repr__(self):
        return repr(self.to_dict())

    def to_dict(self):
        return {f: getattr(self, f) for f in self.non_empty_fields}
