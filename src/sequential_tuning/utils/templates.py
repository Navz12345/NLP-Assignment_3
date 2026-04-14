from __future__ import annotations

from pathlib import Path


def load_template(prompts_dir: str | Path, name: str) -> str:
    template_path = Path(prompts_dir) / name
    return template_path.read_text(encoding="utf-8")

