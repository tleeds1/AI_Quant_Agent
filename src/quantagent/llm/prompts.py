"""llm/prompts.py -- versioned prompt template loader (guideline.md §7:
"Prompts live in prompts/<stage>/<name>.v<N>.jinja, versioned. Never inline
a multi-line prompt in Python. A prompt change is a code change and goes
through review.").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PROMPTS_DIR = _REPO_ROOT / "prompts"


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    """`version` is the exact string recorded in the trace (architecture.md
    §9.1: "a behaviour change is attributable") -- e.g. `"planner/dag.v1"`.
    """

    text: str
    version: str


class PromptLoader:
    """Thin wrapper over a Jinja2 `Environment` scoped to `prompts/`.

    `StrictUndefined` turns a missing context variable into a loud
    `jinja2.UndefinedError` at render time rather than a silently blank
    section -- a prompt template is code, and code with an unset variable
    should fail, not render wrong.
    """

    def __init__(self, prompts_dir: Path | None = None) -> None:
        directory = prompts_dir or _DEFAULT_PROMPTS_DIR
        self._env = Environment(
            loader=FileSystemLoader(str(directory)),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=False,
        )

    def render(self, stage: str, name: str, version: int, /, **context: Any) -> RenderedPrompt:
        """`stage`/`name`/`version` are positional-only: template context
        commonly needs its own `name` key (a tool name, a portfolio name),
        which would otherwise collide with this method's own `name`
        parameter -- caught by `test_prompts.py::
        test_context_name_key_does_not_collide_with_loader_name_param`.
        """
        template_name = f"{stage}/{name}.v{version}.jinja"
        template = self._env.get_template(template_name)
        text = template.render(**context)
        return RenderedPrompt(text=text, version=f"{stage}/{name}.v{version}")
