from __future__ import annotations

from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parent
__package__ = __name__
__path__ = [str(_SOURCE_ROOT)]

exec((_SOURCE_ROOT / "__init__.py").read_text(encoding="utf-8"), globals())
