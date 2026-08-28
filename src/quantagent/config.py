from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process-wide configuration loaded from the environment (guideline.md §2.4)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    market_data_provider: str = "yfinance"
    polygon_api_key: str = ""
    database_url: str = "postgresql+asyncpg://quant:quant@localhost:5432/quantagent"
    redis_url: str = "redis://localhost:6379/0"
    sec_user_agent: str = ""
    model_planner: str = "claude-sonnet-4-6"
    model_synthesizer: str = "claude-sonnet-4-6"
    model_critic: str = "claude-sonnet-4-6"
    model_intent: str = "claude-haiku-4-5"
    max_tool_calls: int = Field(default=12, gt=0)
    max_wall_ms: int = Field(default=12_000, gt=0)
    max_usd_per_request: float = Field(default=0.15, gt=0)
    log_level: str = "INFO"
    env: str = "local"


settings = Settings()
