"""Pytest configuration: ensure the src/ layout is importable from tests/."""

import sys
from pathlib import Path

# Make the installed package (or editable install) available. When running
# directly without `pip install -e .`, we add src/ to sys.path as a fallback.
_src = Path(__file__).parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
