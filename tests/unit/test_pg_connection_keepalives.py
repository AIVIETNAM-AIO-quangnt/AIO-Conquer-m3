"""Regression test for the idle-connection drop that broke ``bronze_to_silver``.

``bronze_to_silver``/``silver_to_gold`` hold a single ``pg_connection()`` open
(idle) across ``ops.track_run`` while the actual bulk transform runs for tens of
minutes on a *separate* Ibis/DuckDB connection. Without TCP keepalives, whatever
sits between here and Postgres (cloud LB, NAT, pooler) can silently drop that idle
connection with no FIN/RST -- the next statement on it then fails as
``OperationalError: SSL SYSCALL error: EOF detected`` instead of a clean,
retryable error. See ``conquer3/db/engine.py``'s ``pg_connection`` docstring.

Mocks ``psycopg2.connect`` rather than needing a real Postgres/Docker, mirroring
``tests/unit/test_duckdb_file_lock.py``'s from-scratch repro style.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from conquer3.config.settings import PgSettings
from conquer3.db.engine import pg_connection


def test_pg_connection_requests_tcp_keepalives() -> None:
    settings = PgSettings(sslmode="disable", channel_binding="disable")

    with patch("conquer3.db.engine.psycopg2.connect") as mock_connect:
        mock_connect.return_value = MagicMock()
        with pg_connection(settings):
            pass

    _, kwargs = mock_connect.call_args
    assert kwargs["keepalives"] == 1
    assert kwargs["keepalives_idle"] > 0
    assert kwargs["keepalives_interval"] > 0
    assert kwargs["keepalives_count"] > 0
