"""The single most load-bearing test in the repo for the Colab path.

``conquer3.core`` is pip-installed into Google Colab with a bare ``pip install .``
(no extras beyond ``train``). If anything in ``core`` -- or in ``conquer3/__init__.py``,
which runs first -- grows an import of pandas/duckdb/pathway/bentoml, the notebook
breaks in a way that only shows up in Colab.

These tests run the import in a *subprocess*, because by the time pytest has imported
the rest of the suite the forbidden modules are already in ``sys.modules``.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

FORBIDDEN = [
    "numpy",
    "pandas",
    "polars",
    "pyarrow",
    "pydantic",
    "pydantic_settings",
    "redis",
    "duckdb",
    "ibis",
    "pathway",
    "bentoml",
    "mlflow",
    "sklearn",
    "joblib",
    "psycopg",
    "opentelemetry",
]

CORE_MODULES = [
    "conquer3",
    "conquer3.core.features",
    "conquer3.core.schema",
    "conquer3.core.serde",
    "conquer3.core.timeref",
    "conquer3.core.types",
    "conquer3.contracts.events",
]


def _import_in_subprocess(module: str) -> set[str]:
    """Import ``module`` in a clean interpreter, return forbidden modules loaded."""
    script = (
        f"import {module}, sys, json\n"
        f"forbidden = {FORBIDDEN!r}\n"
        "loaded = sorted(m for m in forbidden if m in sys.modules)\n"
        "print(json.dumps(loaded))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"importing {module} failed:\n{result.stderr}"
    import json

    return set(json.loads(result.stdout.strip().splitlines()[-1]))


@pytest.mark.parametrize("module", CORE_MODULES)
def test_module_pulls_no_heavy_dependency(module: str) -> None:
    leaked = _import_in_subprocess(module)
    assert not leaked, (
        f"{module} pulled in {sorted(leaked)}. "
        "core/ and contracts.events must stay stdlib-only so Colab can install them."
    )


def test_package_init_defines_only_metadata() -> None:
    """A re-export in __init__.py would defeat every test above.

    Runs in a subprocess like the tests above -- checking ``vars(conquer3)``
    in-process would pick up whatever *other* test modules happened to import
    ``conquer3.db``/``conquer3.pipelines``/etc. earlier in the same pytest session
    (ordinary Python submodule binding, nothing to do with ``__init__.py`` itself),
    which is exactly the cross-test pollution this file's docstring warns about.
    """
    script = (
        "import conquer3, json\n"
        "print(json.dumps(sorted(n for n in vars(conquer3) if not n.startswith('_'))))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, f"importing conquer3 failed:\n{result.stderr}"
    import json

    public = set(json.loads(result.stdout.strip().splitlines()[-1]))
    assert public == set(), (
        f"conquer3/__init__.py exposes {sorted(public)}; it must import nothing."
    )
