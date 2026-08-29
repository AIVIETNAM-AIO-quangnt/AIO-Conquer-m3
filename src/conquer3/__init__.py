"""conquer3 — credit-fraud detection MLOps platform.

This module MUST NOT import anything from the package.

Colab installs this distribution with only the ``train`` extra and then runs
``import conquer3.core.features``, which executes this file first. A single
convenience re-export here (e.g. ``from conquer3.serving import ...``) would drag
mlflow/redis into that import and break the notebook. Keep it to metadata.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
