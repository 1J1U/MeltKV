import json
import pathlib

from .singleton import Singleton


class RecursiveNamespace:
    def __init__(self, dict_=None):
        self.__dict__["_attributes"] = {}
        if dict_ is not None:
            self._attributes = RecursiveNamespace.from_dict(dict_)._attributes

    def __getattr__(self, name):
        if name not in self._attributes:
            self._attributes[name] = RecursiveNamespace()
        return self._attributes[name]

    def __setattr__(self, name, value):
        if isinstance(value, RecursiveNamespace):
            self._attributes[name] = value
        else:
            current = self
            *parts, last = name.split(".")
            for part in parts:
                current = getattr(current, part)
            current._attributes[last] = value

    def __repr__(self):
        return repr(self._attributes)

    def __iadd__(self, other):
        for key, value in other._attributes.items():
            self._attributes[key] = value
        return self

    def to_dict(self):
        return {
            k: v.to_dict() if isinstance(v, RecursiveNamespace) else v
            for k, v in self._attributes.items()
            if not k.startswith("_")
        }

    def to_serializable_dict(self):
        def is_json_serializable(value):
            if isinstance(value, pathlib.Path):
                return False
            try:
                json.dumps(value)
                return True
            except (TypeError, OverflowError):
                return False

        return {
            k: v.to_serializable_dict() if isinstance(v, RecursiveNamespace) else v
            for k, v in self._attributes.items()
            if is_json_serializable(v)
        }

    @staticmethod
    def from_dict(d):
        ns = RecursiveNamespace()
        for k, v in d.items():
            if isinstance(v, dict):
                ns._attributes[k] = RecursiveNamespace.from_dict(v)
            else:
                ns._attributes[k] = v
        return ns

    def hasattr(self, name):
        return name in self._attributes


class MeltKVConfig(RecursiveNamespace, metaclass=Singleton):

    pass


def load_config(file_path):
    with open(file_path, "r") as f:
        return RecursiveNamespace.from_dict(json.loads(f.read()))
