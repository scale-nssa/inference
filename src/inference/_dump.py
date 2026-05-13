from __future__ import annotations

import base64
from typing import Any


def dump_provider_response(response: Any) -> dict[str, Any]:
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        for kwargs in ({"mode": "json"}, {}):
            try:
                dumped = model_dump(**kwargs)
            except TypeError:
                continue
            if isinstance(dumped, dict):
                return _coerce_json(dumped)

    for method_name in ("to_json_dict", "to_dict"):
        method = getattr(response, method_name, None)
        if callable(method):
            dumped = method()
            if isinstance(dumped, dict):
                return _coerce_json(dumped)

    raise TypeError(
        f"provider response {type(response).__name__} could not be serialized"
    )


def _coerce_json(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, dict):
        return {str(key): _coerce_json(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_coerce_json(item) for item in value]

    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(mode="json")
        except TypeError:
            dumped = model_dump()
        return _coerce_json(dumped)

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _coerce_json(to_dict())

    return str(value)
