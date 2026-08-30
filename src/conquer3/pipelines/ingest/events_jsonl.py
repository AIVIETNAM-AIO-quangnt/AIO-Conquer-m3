"""Landing scored events JSONL from the scorer into ``bronze.scored_events``.

Offset-based ingest: tracks bytes_consumed in ops.file_ingest_log to avoid
reprocessing and never passes the last newline (incomplete-line protection).

The scorer writes JSONL with one ScoredEvent per line to `${C3_EVENT_DIR}/scored/dt=…/hr=…/*.jsonl`.
"""

from __future__ import annotations

import json
from pathlib import Path

from conquer3.db.engine import pg_connection

__all__ = ["ingest_events_jsonl"]


def ingest_events_jsonl(conn) -> int:  # type: ignore
    """Consume JSONL from scored event files and insert into bronze.scored_events.

    Offset-based: reads ops.file_ingest_log to find the last consumed position
    and only processes new lines. Never passes the last incomplete line.

    Args:
        conn: psycopg connection (used for bulk inserts)

    Returns:
        Number of rows inserted.
    """
    from conquer3.config.settings import get_settings

    settings = get_settings()
    event_dir = Path(settings.event.dir) / "scored"

    rows_inserted = 0

    # Find all JSONL files in the scored events directory
    if not event_dir.is_dir():
        # No events yet; this is ok (scorer may not have started)
        return 0

    jsonl_files = sorted(event_dir.glob("dt=*/hr=*/*.jsonl"))

    for file_path in jsonl_files:
        # Track this file's position
        file_key = str(file_path.relative_to(event_dir))

        # Get last consumed byte position for this file
        with conn.cursor() as cur:
            cur.execute(
                "SELECT bytes_consumed FROM ops.file_ingest_log "
                "WHERE file_path = %s ORDER BY consumed_at DESC LIMIT 1",
                (file_key,),
            )
            result = cur.fetchone()
            start_byte = result[0] if result else 0

        # Read from start_byte onwards, but stop at the last complete line
        if not file_path.is_file():
            continue

        with open(file_path, "rb") as f:
            f.seek(start_byte)
            content = f.read()

        # Find the position of the last complete newline
        last_newline = content.rfind(b"\n")
        if last_newline == -1:
            # No complete lines in this chunk
            continue

        # Only process up to (and including) the last newline
        complete_content = content[: last_newline + 1]
        consumed_bytes = start_byte + len(complete_content)

        # Parse and insert each line
        lines = complete_content.decode("utf-8").strip().split("\n")
        batch_payloads = []
        for line in lines:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                # Validate required fields
                if not payload.get("event_id"):
                    print(f"Skipped event with missing event_id in {file_path}")
                    continue
                batch_payloads.append((payload["event_id"], json.dumps(payload)))
            except (json.JSONDecodeError, Exception) as e:
                # Log and continue (skip malformed lines)
                print(f"Skipped malformed line in {file_path}: {e}")
                continue

        # Bulk insert all payloads
        if batch_payloads:
            with conn.cursor() as cur:
                for event_id, payload_str in batch_payloads:
                    cur.execute(
                        "INSERT INTO bronze.scored_events (event_id, payload) "
                        "VALUES (%s, %s) ON CONFLICT (event_id) DO NOTHING",
                        (event_id, payload_str),
                    )
                conn.commit()
            rows_inserted += len(batch_payloads)

        # Record progress in ops.file_ingest_log
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO ops.file_ingest_log (file_path, bytes_consumed) "
                "VALUES (%s, %s) ON CONFLICT (file_path) DO UPDATE SET "
                "bytes_consumed = EXCLUDED.bytes_consumed, consumed_at = now()",
                (file_key, consumed_bytes),
            )
        conn.commit()

    return rows_inserted
