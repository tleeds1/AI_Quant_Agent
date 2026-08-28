from __future__ import annotations

from pathlib import Path

import jinja2
import pytest

from quantagent.llm.prompts import PromptLoader


@pytest.fixture
def prompts_dir(tmp_path: Path) -> Path:
    stage_dir = tmp_path / "greet"
    stage_dir.mkdir()
    (stage_dir / "hello.v1.jinja").write_text("Hello, {{ name }}!", encoding="utf-8")
    return tmp_path


def test_render_returns_text_and_version(prompts_dir: Path) -> None:
    loader = PromptLoader(prompts_dir)

    rendered = loader.render("greet", "hello", 1, name="World")

    assert rendered.text == "Hello, World!"
    assert rendered.version == "greet/hello.v1"


def test_missing_template_raises(prompts_dir: Path) -> None:
    loader = PromptLoader(prompts_dir)

    with pytest.raises(jinja2.TemplateNotFound):
        loader.render("greet", "does_not_exist", 1, name="World")


def test_missing_context_variable_raises_strict_undefined(prompts_dir: Path) -> None:
    loader = PromptLoader(prompts_dir)

    with pytest.raises(jinja2.UndefinedError):
        loader.render("greet", "hello", 1)


def test_context_name_key_does_not_collide_with_loader_name_param(prompts_dir: Path) -> None:
    loader = PromptLoader(prompts_dir)

    rendered = loader.render("greet", "hello", 1, name="Portfolio Name")

    assert rendered.text == "Hello, Portfolio Name!"


def test_real_prompts_dir_defaults_to_repo_prompts_directory() -> None:
    loader = PromptLoader()
    # Smoke test only: constructing against the real prompts/ directory
    # (populated later in this milestone) must not raise.
    assert loader is not None
