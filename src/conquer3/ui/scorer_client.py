"""Thin HTTP client over the scorer's ``POST /predict`` and ``POST /model_info``.

``conquer3.ui`` must never import ``conquer3.serving`` (see the "ui talks to
serving over HTTP, never by import" import-linter contract) -- this module talks
to the scorer exactly as any external caller would: plain JSON in, plain JSON
out. No serving-side pydantic model is imported here.
"""

from __future__ import annotations

from typing import Any

import httpx

__all__ = ["ScorerError", "get_model_info", "is_scorer_healthy", "score_transactions"]


class ScorerError(RuntimeError):
    """Raised when the scorer returns a non-2xx response."""


def score_transactions(
    transactions: list[dict[str, Any]],
    *,
    base_url: str,
    dry_run: bool = False,
    batch_size: int = 500,
    timeout_s: float = 60.0,
) -> list[dict[str, Any]]:
    """POSTs ``transactions`` to ``{base_url}/predict`` in batches of
    ``batch_size``, returning one ``ScoreResult``-shaped dict per input row, in
    the same order they were submitted.
    """
    url = base_url.rstrip("/") + "/predict"
    results: list[dict[str, Any]] = []
    with httpx.Client(timeout=timeout_s) as client:
        for start in range(0, len(transactions), batch_size):
            batch = transactions[start : start + batch_size]
            resp = client.post(url, json={"transactions": batch, "dry_run": dry_run})
            if resp.status_code != 200:
                raise ScorerError(f"POST {url} failed with {resp.status_code}: {resp.text[:500]}")
            results.extend(resp.json())
    return results


def get_model_info(*, base_url: str, timeout_s: float = 10.0) -> dict[str, Any]:
    """The champion currently served, from ``POST /model_info``."""
    url = base_url.rstrip("/") + "/model_info"
    resp = httpx.post(url, json={}, timeout=timeout_s)
    if resp.status_code != 200:
        raise ScorerError(f"POST {url} failed with {resp.status_code}: {resp.text[:500]}")
    result: dict[str, Any] = resp.json()
    return result


def is_scorer_healthy(*, base_url: str, timeout_s: float = 5.0) -> bool:
    """Sidebar health indicator -- BentoML's own ``/readyz``, which is 500 until
    the champion has finished loading (see ``docker-compose.yaml``'s ``scorer``
    healthcheck)."""
    url = base_url.rstrip("/") + "/readyz"
    try:
        return httpx.get(url, timeout=timeout_s).status_code == 200
    except httpx.HTTPError:
        return False
