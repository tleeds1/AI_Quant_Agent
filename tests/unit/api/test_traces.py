from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from quantagent.data.repositories.trace_repository import TraceRepository


def test_get_trace_success(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    expected_trace = {
        "id": "tr_123",
        "tenant_id": "tenant_abc",
        "question": "test question",
        "answer": {"decision": "HOLD"},
    }

    mock_get_trace = AsyncMock(return_value=expected_trace)
    monkeypatch.setattr(TraceRepository, "get_trace", mock_get_trace)

    response = client.get("/v1/traces/tr_123", headers={"X-Tenant-Id": "tenant_abc"})

    assert response.status_code == 200
    assert response.json() == expected_trace
    mock_get_trace.assert_called_once_with("tr_123", tenant_id="tenant_abc")


def test_get_trace_not_found(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    mock_get_trace = AsyncMock(return_value=None)
    monkeypatch.setattr(TraceRepository, "get_trace", mock_get_trace)

    response = client.get("/v1/traces/tr_missing", headers={"X-Tenant-Id": "tenant_abc"})

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_get_trace_missing_tenant_header(client: TestClient) -> None:
    response = client.get("/v1/traces/tr_123")

    assert response.status_code == 400
    assert "Tenant-Id" in response.json()["detail"]
