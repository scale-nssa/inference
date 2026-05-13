from __future__ import annotations

import json
import urllib.error
import urllib.request

LM_STUDIO_BASE = "http://localhost:1234"


def ensure_model_loaded(model_name: str) -> None:
    """Verify the requested model is loaded in LM Studio; load it if not.

    Raises ConnectionError if LM Studio is not reachable.
    Raises ValueError if the model is not available in LM Studio.
    """
    models = _list_models()
    available = {entry["key"]: entry for entry in models}

    if model_name in available and available[model_name].get("loaded_instances"):
        return

    if model_name not in available:
        raise ValueError(
            f"model {model_name!r} not found in LM Studio. "
            f"available: {sorted(available)}"
        )

    for entry in models:
        for instance in entry.get("loaded_instances", []):
            _unload_model(instance["id"])

    print(f"[lm-studio] loading {model_name}...")
    _load_model(model_name)
    print(f"[lm-studio] loaded {model_name}")


def _list_models() -> list[dict]:
    try:
        with urllib.request.urlopen(
            f"{LM_STUDIO_BASE}/api/v1/models", timeout=10
        ) as response:
            payload = json.loads(response.read())
    except urllib.error.URLError as exc:
        raise ConnectionError(
            f"LM Studio not reachable at {LM_STUDIO_BASE}: {exc.reason}"
        ) from exc
    return payload["models"]


def _unload_model(instance_id: str) -> None:
    request = urllib.request.Request(
        f"{LM_STUDIO_BASE}/api/v1/models/unload",
        data=json.dumps({"instance_id": instance_id}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        response.read()


def _load_model(model_name: str) -> None:
    request = urllib.request.Request(
        f"{LM_STUDIO_BASE}/api/v1/models/load",
        data=json.dumps({"model": model_name}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        response.read()
