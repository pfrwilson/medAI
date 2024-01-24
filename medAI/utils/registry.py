from dataclasses import dataclass
from simple_parsing import parse, subgroups
from typing import Any
import typing
import inspect


class Registry:
    """A registry for easily creating and managing classes and their configurations.
    For an example, see medAI/projects/sam/medsam_cancer_detection_model_registry.py
    """

    def __init__(self, name: str = "registry", help: str = "A registry of classes."):
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
            defaults = [None] * (n_fields - n_defaults) + list(
                cls.__init__.__defaults__
            )

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
            {name: config for name, (_, config) in self._registry.items()},
            default=next(iter(self._registry.keys())),
        )


class FactoryRegistry:
    """A registry for easily creating and managing classes and their configurations.
    
    Args: 
        name: The name of this registry.
        desc: A description of this registry. 

    Example:
        >>> registry = FactoryRegistry("Model", "Model used in the experiment.")
        >>> @registry.register
        >>> def model_a(layers: int = 3):
        >>>     return Model(layers=layers)
        >>> @registry.register
        >>> def model_b(layers: int = 5, dropout: float = 0.1):
        >>>     return Model(layers=layers)
        >>> parser = ArgumentParser()
        >>> registry.add_argparse_args(parser)
        >>> args = parser.parse_args()
        >>> model = registry.build_object_from_argparse_args(args)
    """

    def __init__(self, name, desc=None):
        self.name = name
        self._registry = {}
        self._params = {}
        self.help = desc

    def register(self, name=None):
        def wrapper(fn):
            if name is None:
                _name = fn.__name__
            else:
                _name = name
            self._registry[_name] = fn
            self._params[_name] = inspect.signature(fn).parameters
            return fn
        return wrapper

    def add_argparse_args(self, parser, prefix="", conflict_handler="raise"):
        """Add the argparse arguments for this registry to the given parser.
        
        First, the choice of factory is added as an argument. Then, the arguments for the
        chosen factory are added as arguments to the parser. This way the user can choose
        which factory to use, and then configure the factory using the arguments.

        Args:
            parser: The parser to add the arguments to.
        """
        if conflict_handler != "raise":
            raise NotImplementedError(
                "Only conflict_handler='raise' is currently supported."
            )

        group = parser.add_argument_group(self.name, self.help)
        group.add_argument(
            f"--{self.name.lower()}",
            type=str,
            default=list(self._registry.keys())[0],
            choices=self._registry.keys(),
            help="choice of factory",
        )
        args, _ = parser.parse_known_args()

        factory_name_choice = args.__dict__[f"{self.name.lower()}"]
        if factory_name_choice is None:
            return

        params = self._params[factory_name_choice]
        for param_name, param in params.items():
            type = (
                param.annotation
                if param.annotation is not inspect.Parameter.empty
                else None
            )
            if typing.get_origin(type) == typing.Literal: 
                choices = typing.get_args(type)
                type = None # should be obvious from the choices
            else: 
                choices = None
            default = (
                param.default if param.default is not inspect.Parameter.empty else None
            )
            group.add_argument(
                f"--{prefix}{param_name.lower().replace('_', '-')}",
                type=type,
                default=default,
                help="%(type)s %(default)s",
                choices=choices,
            )

    def add_simple_parsing_subgroups(self, config):
        ...

    def build_object_from_argparse_args(self, args):
        factory_name_choice = args.__dict__[f"{self.name.lower()}"]
        if factory_name_choice is None:
            return None

        params = self._params[factory_name_choice]
        kwargs = {
            param_name: args.__dict__[f"{param_name}"]
            for param_name, param in params.items()
        }
        return self.build_object(factory_name_choice, **kwargs)

    def get_factory(self, name):
        return self._registry[name]

    def get_args(self, name):
        return self._params[name]

    def build_object(self, name, **kwargs):
        return self._registry[name](**kwargs)


