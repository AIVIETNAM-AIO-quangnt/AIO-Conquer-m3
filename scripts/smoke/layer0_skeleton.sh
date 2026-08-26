#!/usr/bin/env bash
# Layer 0 gate: repo skeleton, import boundaries, and the Colab install path.
# Do not start Layer 1 (infra) until this passes.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

echo "== uv sync --all-extras =="
uv sync --all-extras

echo "== ruff =="
uv run ruff check .
uv run ruff format --check .

echo "== mypy =="
uv run mypy

echo "== import-linter (the dependency-boundary contracts) =="
uv run lint-imports

echo "== pytest (includes the subprocess dependency-light checks) =="
uv run pytest tests/

echo "== conquer3 --help =="
uv run conquer3 --help >/dev/null
uv run conquer3 version

echo "== bare-venv install: the actual Colab path =="
# `pip install .` with --no-deps, exactly as Colab's
# `pip install "conquer3[train] @ git+..."` would resolve `conquer3` itself, minus
# the train extra's own dependencies (those are expected -- pandas/sklearn are fine
# in Colab, they just must not be required to import conquer3.core).
TMP_VENV="$(mktemp -d)/venv"
uv venv "$TMP_VENV" --python 3.12 -q
VIRTUAL_ENV="$TMP_VENV" uv pip install -q --no-deps .
INSTALLED="$(VIRTUAL_ENV="$TMP_VENV" uv pip list 2>/dev/null | tail -n +3 | awk '{print $1}')"
if [ "$INSTALLED" != "conquer3" ]; then
  echo "FAIL: bare install pulled in more than conquer3: $INSTALLED" >&2
  exit 1
fi
"$TMP_VENV/bin/python" -c "
import sys
import conquer3.core.features
import conquer3.core.schema
import conquer3.core.serde
import conquer3.core.timeref
import conquer3.core.types
import conquer3.contracts.events
heavy = {'numpy','pandas','pydantic','sklearn','mlflow','redis','duckdb','pathway','bentoml'}
loaded = heavy & set(sys.modules)
assert not loaded, f'heavy deps leaked into a bare install: {loaded}'
print('bare-venv Colab-path import: OK')
"
rm -rf "$(dirname "$TMP_VENV")"

echo
echo "Layer 0 gate: PASS"
