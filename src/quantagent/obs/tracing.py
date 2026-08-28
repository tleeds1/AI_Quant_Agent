from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider


def configure_tracing() -> None:
    """Install the OTel tracer provider. No exporter is wired yet — spans are
    created but not shipped anywhere until M6 adds an OTLP exporter
    (architecture.md §9). Idempotent.
    """
    if isinstance(trace.get_tracer_provider(), TracerProvider):
        return
    provider = TracerProvider(resource=Resource.create({SERVICE_NAME: "quantagent"}))
    trace.set_tracer_provider(provider)
