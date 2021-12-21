class RepresentMixin:
    def __repr__(self):
        return repr(self.to_dict())

    def to_dict(self):
        return {f: v for f, v in self}
