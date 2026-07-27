import sys
from pathlib import Path

# Tests import the project's packages directly; make the repo root importable
# without requiring an editable install in CI.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
