"""misc.py"""


from typing import Any, Optional, Type


def validate_type(var: Any, type_: Type) -> bool:
    if not isinstance(var, type_):
        raise TypeError(f"Expected type is `{type_.__name__}`, but got `{type(var).__name__}`")
    
    return True


def get_item(dict_: dict, key: Any, throw_exception: bool = True) -> Optional[Any]:
    validate_type(dict_, dict)

    item = dict_.get(key)

    if throw_exception and item is None:
        raise KeyError(f"Expected key is {key}, but not found")

    return item
