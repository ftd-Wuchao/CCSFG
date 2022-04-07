# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from ...utils.registry import Registry

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for datasets
It must returns an instance of :class:`Backbone`.
"""

# Person re-id datasets
from .market1501 import Market1501
from .msmt17 import MSMT17

__all__ = [k for k in globals().keys() if "builtin" not in k and not k.startswith("_")]
