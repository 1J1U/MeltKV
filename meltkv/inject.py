

from typing import Callable


class MethodInjector:
    def __init__(self, node, attr_name: str, other_attr):
        self.node = node
        self.attr_name = attr_name
        self._attr = getattr(node, attr_name, None)
        self.other_attr = other_attr

    def __enter__(self):
        setattr(self.node, self.attr_name, self.other_attr)
        return self.node

    def __exit__(self, exc_type, exc_value, traceback):
        if self._attr is None:
            delattr(self.node, self.attr_name)
        else:
            setattr(self.node, self.attr_name, self._attr)
        return False


class ContextList(list):
    def __enter__(self):
        for item in self:
            item.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for item in reversed(self):
            item.__exit__(exc_type, exc_value, traceback)
        return False
