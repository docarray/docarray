from .property import PropertyMixin
from .representer import RepresentMixin


class AllMixins(RepresentMixin, PropertyMixin):
    ...
