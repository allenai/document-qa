import json
from collections import OrderedDict
from inspect import signature
from warnings import warn

import numpy as np
from sklearn.base import BaseEstimator


class Configuration(object):
    def __init__(self, name, version, params):
        if not isinstance(name, str):
            raise ValueError()
        if not isinstance(params, dict):
            raise ValueError()
        self.name = name
        self.version = version
        self.params = params

    def __str__(self):
        if len(self.params) == 0:
            return "%s-v%s" % (self.name, self.version)
        json_params = config_to_json(self.params)
        if len(json_params) < 200:
            return "%s-v%s: %s" % (self.name, self.version, json_params)
        else:
            return "%s-v%s {...}" % (self.name, self.version)

    def __eq__(self, other):
        return isinstance(other, Configuration) and \
               self.name == other.name and \
               self.version == other.version and \
               self.params == other.params


class Configurable(object):
    """
    Configurable classes have names, versions, and a set of parameters that are either "simple" aka JSON serializable
    types or other Configurable objects. Configurable objects should also be serializable via pickle.
    Configurable classes are defined mainly to give us a human-readable way of reading of the `parameters`
    set for different objects and to attach version numbers to them.

    By default we follow the format sklearn uses for its `BaseEstimator` class, where parameters are automatically
    derived based on the constructor parameters.
    """

    @classmethod
    def _get_param_names(cls):
        # fetch the constructor or the original constructor before
        init = cls.__init__
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        init_signature = signature(init)
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self']
        if any(p.kind == p.VAR_POSITIONAL for p in parameters):
            raise RuntimeError()
        return sorted([p.name for p in parameters])

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def version(self):
        return 0

    def get_params(self):
        out = {}
        for key in self._get_param_names():
            v = getattr(self, key, None)
            if isinstance(v, Configurable):
                out[key] = v.get_config()
            elif hasattr(v, "get_config"):  # for keras objects
                out[key] = {"name": v.__class__.__name__, "config": v.get_config()}
            else:
                out[key] = v
        return out

    def get_config(self) -> Configuration:
        params = {k: describe(v) for k,v in self.get_params().items()}
        return Configuration(self.name, self.version, params)

    def __getstate__(self):
        state = dict(self.__dict__)
        if "version" in state:
            if state["version"] != self.version:
                raise RuntimeError()
        else:
            state["version"] = self.version
        return state

    def __setstate__(self, state):
        if "version" not in state:
            raise RuntimeError("Version should be in state (%s)" % self.__class__.__name__)
        if state["version"] != self.version:
            warn(("%s loaded with version %s, but class " +
                 "version is %s") % (self.__class__.__name__, state["version"], self.version))

        if "state" in state:
            self.__dict__ = state["state"]
        else:
            del state["version"]
            self.__dict__ = state


def describe(obj):
    if isinstance(obj, Configurable):
        return obj.get_config()
    else:
        obj_type = type(obj)

        if obj_type in (list, set, frozenset, tuple):
            return obj_type([describe(e) for e in obj])
        elif isinstance(obj, tuple):
            # Name tuple, convert to tuple
            return tuple(describe(e) for e in obj)
        elif obj_type in (dict, OrderedDict):
            output = OrderedDict()
            for k, v in obj.items():
                if isinstance(k, Configurable):
                    raise ValueError()
                output[k] = describe(v)
            return output
        else:
            return obj


class EncodeDescription(json.JSONEncoder):
    """ Json encoder that encodes 'Configurable' objects as dictionaries and handles
    some numpy types. Note decoding this output will not reproduce the original input,
    for these types, this is only intended to be used to produce human readable output.
    '"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.dtype):
            return str(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, BaseEstimator):  # handle sklearn estimators
            return Configuration(obj.__class__.__name__, 0, obj.get_params())
        elif isinstance(obj, Configuration):
            if "version" in obj.params or "name" in obj.params:
                raise ValueError()
            out = OrderedDict()
            out["name"] = obj.name
            if obj.version != 0:
                out["version"] = obj.version
            out.update(obj.params)
            return out
        elif isinstance(obj, Configurable):
            return obj.get_config()
        elif isinstance(obj, set):
            return sorted(obj)  # Ensure deterministic order
        else:
            try:
                return super().default(obj)
            except TypeError:
                return str(obj)


def config_to_json(data, indent=None):
    return json.dumps(data, sort_keys=False, cls=EncodeDescription, indent=indent)
