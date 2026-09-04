"""
Prompts live next to this file as plain .txt so they can be tuned without
touching Python. Load one with `load("corrector")`.

Placeholders are filled with str.replace, not str.format, because OCR text
regularly contains braces that would crash a format call.
"""
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent


def load(name):
    """Read prompts/<name>.txt."""
    path = PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"No prompt template at {path}")
    return path.read_text(encoding="utf-8").strip()


def render(name, **placeholders):
    """Load a template and substitute {key} placeholders literally."""
    text = load(name)
    for key, value in placeholders.items():
        text = text.replace("{" + key + "}", str(value))
    return text
