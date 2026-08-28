from __future__ import annotations

import inspect
import json

import pytest
from pydantic import BaseModel

from quantagent.contracts import tools as tools_module

_ALL_MODELS = [
    cls
    for _, cls in inspect.getmembers(tools_module, inspect.isclass)
    if issubclass(cls, BaseModel) and cls.__module__ == tools_module.__name__
]


@pytest.mark.parametrize("model", _ALL_MODELS, ids=lambda m: m.__name__)
def test_model_json_schema_is_valid_and_serializable(model: type[BaseModel]) -> None:
    schema = model.model_json_schema()

    assert isinstance(schema, dict)
    # A valid JSON Schema round-trips through json.dumps/loads without error.
    assert json.loads(json.dumps(schema)) == schema


def test_every_tools_module_model_was_discovered() -> None:
    # Guards the discovery mechanism itself: if inspect.getmembers ever
    # returned nothing, every other test in this file would vacuously pass.
    assert len(_ALL_MODELS) >= 40
