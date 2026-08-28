from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest
from fastapi.testclient import TestClient

import quantagent.api.routes.analyze as analyze_module
from quantagent.agent.events import (
    DraftEvent,
    FinalEvent,
    PlanEvent,
    ToolDoneEvent,
    ToolStartEvent,
    VerdictEvent,
)
from quantagent.api.app import create_app
from quantagent.api.deps import get_app_resources
from quantagent.contracts.answer import AgentAnswer
from quantagent.contracts.verification import VerificationReport


def _answer() -> AgentAnswer:
    return AgentAnswer(
        trace_id="tr_1",
        scope="PORTFOLIO",
        decision="HOLD",
        confidence=0.5,
        confidence_basis=[],
        risk_level="LOW",
        horizon="n/a",
        summary="s",
        claims=[],
        evidence=[],
        quant_metrics={},
        constraints_checked=[],
        limitations=["none"],
        disclosures=[],
        verification=VerificationReport(verdict="PASS", checks=0, warnings=0, repair_attempts=0),
    )


class _RaisingResources:
    def tool_context(self, tenant_id: str) -> None:
        raise AssertionError("tool_context must not be called before header validation")


class _FakeResources:
    anthropic_client = object()
    prompt_loader = object()

    def tool_context(self, tenant_id: str) -> object:
        return object()


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    app.dependency_overrides[get_app_resources] = lambda: _RaisingResources()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_missing_tenant_header_returns_400_before_touching_resources(client: TestClient) -> None:
    response = client.post("/v1/analyze", json={"question": "hi"})
    assert response.status_code == 400


def test_empty_question_returns_422(client: TestClient) -> None:
    response = client.post("/v1/analyze", json={"question": ""}, headers={"X-Tenant-Id": "t1"})
    assert response.status_code == 422


def _fixed_event_sequence() -> list[object]:
    return [
        PlanEvent(steps=[{"id": "s1"}]),
        ToolStartEvent(call_id="tc_1", tool="get_holdings"),
        ToolDoneEvent(call_id="tc_1", latency_ms=10, status="OK"),
        DraftEvent(answer=_answer()),
        VerdictEvent(verdict="PASS", warnings=0, repair_attempts=0),
        FinalEvent(answer=_answer()),
    ]


async def _fake_loop(*args: object, **kwargs: object) -> AsyncIterator[object]:
    for event in _fixed_event_sequence():
        yield event


def test_sse_framing_round_trips_the_full_event_sequence(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(analyze_module, "run_agent_loop", _fake_loop)
    client.app.dependency_overrides[get_app_resources] = lambda: _FakeResources()

    with client.stream(
        "POST", "/v1/analyze", json={"question": "what's my risk?"}, headers={"X-Tenant-Id": "t1"}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = b"".join(response.iter_bytes())

    frames = [f for f in body.split(b"\n\n") if f.strip()]
    assert len(frames) == 6

    expected_names = ["plan", "tool_start", "tool_done", "draft", "verdict", "final"]
    for frame, expected_name in zip(frames, expected_names, strict=True):
        text = frame.decode()
        assert text.startswith(f"event: {expected_name}\ndata: ")
        payload = json.loads(text.split("data: ", 1)[1])
        assert payload["event"] == expected_name

    final_payload = json.loads(frames[-1].decode().split("data: ", 1)[1])
    AgentAnswer.model_validate(final_payload["answer"])  # must parse as schema-valid


def test_exception_before_loop_starts_still_ends_in_one_final_frame(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _BrokenResources:
        def tool_context(self, tenant_id: str) -> None:
            raise RuntimeError("boom")

    client.app.dependency_overrides[get_app_resources] = lambda: _BrokenResources()

    with client.stream(
        "POST", "/v1/analyze", json={"question": "q"}, headers={"X-Tenant-Id": "t1"}
    ) as response:
        body = b"".join(response.iter_bytes())

    frames = [f for f in body.split(b"\n\n") if f.strip()]
    assert len(frames) == 1
    assert frames[0].startswith(b"event: final\n")
    payload = json.loads(frames[0].decode().split("data: ", 1)[1])
    AgentAnswer.model_validate(payload["answer"])
