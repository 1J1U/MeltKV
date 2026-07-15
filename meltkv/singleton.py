from abc import ABCMeta


class Singleton(type):


    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

    @classmethod
    def has_instance(cls):
        return len(cls._instances) > 0

    @classmethod
    def clear_instance(cls):
        cls._instances = {}


class SingletonABCMeta(Singleton, ABCMeta):
    pass


class NamedSingleton(type):


    _instances = {}

    def __call__(cls, name, *args, **kwargs):
        if name not in cls._instances:
            cls._instances[name] = super().__call__(name, *args, **kwargs)
        return cls._instances[name]
