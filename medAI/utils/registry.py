from dataclasses import dataclass
from simple_parsing import parse, subgroups
from typing import Any


class Registry: 
    """A registry for easily creating and managing classes and their configurations.
    For an example, see medAI/projects/sam/medsam_cancer_detection_model_registry.py
    """

    def __init__(self, name: str = 'registry', help: str = "A registry of classes."): 
        self._registry = {}
        self.name = name
        self.help = help

    @staticmethod
    def add_config(cls):
        n_fields = len(cls.__init__.__annotations__)
        if cls.__init__.__defaults__ is None: 
            n_defaults = 0
            defaults = [None] * n_fields
        else: 
            n_defaults = len(cls.__init__.__defaults__)
            defaults = [None] * (n_fields - n_defaults) + list(cls.__init__.__defaults__)

        class Config: 
            ...

        Config.__doc__ = f"Configuration for {cls.__name__}"
        Config.__annotations__ = cls.__init__.__annotations__
        for name, default in zip(cls.__init__.__annotations__.keys(), defaults):
            if default is not None: 
                setattr(Config, name, default)
        Config.__annotations__["__name__"] = str
        setattr(Config, "__name__", cls.__name__)
        Config = dataclass(Config)

        return Config

    def __call__(self, cls):
        self._registry[cls.__name__] = (cls, self.add_config(cls))
        return cls

    def build_from_config(self, config):
        if isinstance(config, dict): 
            _config_dict = config
        else: 
            _config_dict = config.__dict__.copy()
        name = _config_dict.pop("__name__")
        cls, config_cls = self._registry[name]
        return cls(**_config_dict)
    
    def get_config(self, name: str):
        return self._registry[name][1]
    
    def list_configs(self):
        return list(self._registry.keys())
    
    def get_simple_parsing_subgroups(self):
        return subgroups(
            {
                name: config for name, (_, config) in self._registry.items()
            }, default=next(iter(self._registry.keys()))
        )
