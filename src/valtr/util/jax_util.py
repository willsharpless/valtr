from typing import Any, Callable, ParamSpec, Sequence, TypeVar, Union

import jax
import jax.numpy as jnp
import numpy as np

_PyTree = TypeVar("_PyTree")
_P = ParamSpec("_P")
_R = TypeVar("_R")
_Fn = Callable[_P, _R]

Arr = Union[np.ndarray, jnp.ndarray]


def rep_vmap(fn: _Fn, rep: int, in_axes: int | Sequence[Any] = 0, **kwargs) -> _Fn:
    for ii in range(rep):
        fn = jax.vmap(fn, in_axes=in_axes, **kwargs)
    return fn
