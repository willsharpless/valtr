from dataclasses import MISSING, dataclass, field as _dc_field


def define(_cls=None, *, slots=False, frozen=False, **kwargs):
    def wrap(cls):
        return dataclass(cls, slots=slots, frozen=frozen)
    return wrap if _cls is None else wrap(_cls)


def frozen(_cls=None, **kwargs):
    return define(_cls, frozen=True, **kwargs)


def field(*, eq=True, default=MISSING, default_factory=MISSING, **kwargs):
    params = {"compare": eq}
    if default is not MISSING:
        params["default"] = default
    if default_factory is not MISSING:
        params["default_factory"] = default_factory
    return _dc_field(**params)
