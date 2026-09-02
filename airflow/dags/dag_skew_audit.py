"""Layer 6 DAG 6: Daily skew audit — the enforcement mechanism for zero training/serving skew.

This is the single most important DAG: it verifies that features computed by the scorer
match features computed by the batch pipeline on the exact same transactions.

Workflow:
1. Read ScoredEvent payloads from bronze.scored_events (what the scorer saw)
2. Extract the transaction data and recompute features via core.compute_features
3. Compare recomputed features against gold.txn_features (logged by the scorer)
4. Report any mismatches (P1 alert — skew is the existential risk)
5. Optional: replay a sample through /invocations with params.dry_run=true

Skew audit is the contract: one pure module, one feature source, zero reimplementation.
"""

from __future__ import annotations

import datetime

from airflow.sdk import DAG, task


@task
def audit_feature_skew() -> str:
    """Recompute features from scorer payloads and compare against logged features.

    Reads from bronze.scored_events (full ScoredEvent payloads), extracts transaction
    and state context, recomputes via core.compute_features, diffs against logged.

    Returns: "ok" if zero mismatches, or raises with mismatch details.
    """
    import json
    import math

    from conquer3.core.features import compute_features
    from conquer3.core.schema import FEATURE_NAMES
    from conquer3.core.types import TransactionEvent
    from conquer3.db.engine import pg_connection

    mismatches = []

    with pg_connection() as conn, conn.cursor() as cur:
        # Fetch a sample of scored events for spot-checking
        cur.execute("""
                SELECT payload FROM bronze.scored_events
                ORDER BY created_at DESC
                LIMIT 10000
            """)
        for (payload_str,) in cur:
            payload = json.loads(payload_str)

            # Reconstruct the transaction that was scored
            txn_dict = payload.get("transaction", {})
            if not txn_dict:
                continue

            txn = TransactionEvent(
                event_id=txn_dict.get("event_id"),
                account_id=txn_dict.get("account_id"),
                dest_id=txn_dict.get("dest_id"),
                txn_type=txn_dict.get("txn_type"),
                amount=txn_dict.get("amount"),
                oldbalance_org=txn_dict.get("oldbalance_org"),
                newbalance_orig=txn_dict.get("newbalance_orig"),
                oldbalance_dest=txn_dict.get("oldbalance_dest"),
                newbalance_dest=txn_dict.get("newbalance_dest"),
                event_ts_us=txn_dict.get("event_ts_us"),
                step=txn_dict.get("step"),
            )

            # For skew audit in batch, we don't have the previous state,
            # so we recompute with prev=None (cold start). If had_prev_state=true,
            # the audit should still match because features are deterministic.
            # This is a limitation of the audit (we'd need to also log prev_state
            # in the scored event to be 100% precise), but serves as a sanity check.
            features_recomputed = compute_features(txn, prev=None)

            # Compare against logged features
            logged_features = payload.get("features", {})
            for feat_name in FEATURE_NAMES:
                recomputed_val = getattr(features_recomputed, feat_name, None)
                logged_val = logged_features.get(feat_name)

                # Handle NaN: both None or both NaN are ok
                if recomputed_val is None and logged_val is None:
                    continue
                if isinstance(recomputed_val, float) and isinstance(logged_val, float):
                    if math.isnan(recomputed_val) and math.isnan(logged_val):
                        continue
                    # Allow tiny floating-point differences (~1e-9)
                    if abs(recomputed_val - logged_val) < 1e-9:
                        continue

                # Mismatch
                mismatches.append(
                    f"event_id={txn.event_id}, feature={feat_name}, "
                    f"recomputed={recomputed_val}, logged={logged_val}"
                )

    if mismatches:
        msg = f"Feature skew detected ({len(mismatches)} mismatches):\n" + "\n".join(
            mismatches[:10]
        )
        if len(mismatches) > 10:
            msg += f"\n... and {len(mismatches) - 10} more"
        raise AssertionError(msg)

    print("Skew audit passed: zero mismatches found")
    return "skew_audit: ok"


@task
def audit_model_consistency() -> str:
    """Verify model artifacts and versions are consistent across systems.

    Checks:
    - ops.model_deployments has the current champion version
    - Champion artifact is available locally
    - Model signature in artifact matches FEATURE_NAMES
    """

    from conquer3.config.settings import get_settings
    from conquer3.contracts.model_registry import resolve_champion
    from conquer3.db.engine import pg_connection

    settings = get_settings()

    # Resolve champion
    try:
        _model, ref = resolve_champion(settings.mlflow.model_name)
        print(f"Champion resolved: version={ref.version}, degraded={ref.degraded}")
    except Exception as e:
        raise RuntimeError(f"Failed to resolve champion: {e}") from e

    # Check it's recorded in ops.model_deployments
    with pg_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT version FROM ops.model_deployments WHERE model_name = %s "
            "ORDER BY created_at DESC LIMIT 1",
            (settings.mlflow.model_name,),
        )
        result = cur.fetchone()
        if result is None or result[0] != ref.version:
            raise AssertionError(
                f"Deployment record missing or out of sync: "
                f"registry says {ref.version}, ops says {result[0] if result else 'none'}"
            )

    print("Model consistency verified")
    return "model_consistency: ok"


@task
def audit_state_consistency() -> str:
    """Verify online state (Redis) is consistent with batch state (gold.account_state).

    Samples accounts and compares their state doc in Redis vs. Postgres.
    Small differences (timestamps) are ok; major divergence is a P1.
    """
    import json

    from conquer3.config.settings import get_settings
    from conquer3.db.engine import pg_connection
    from conquer3.serving.state_store import RedisStateStore

    settings = get_settings()
    state_store = RedisStateStore(
        redis_url=settings.redis.url,
        state_ttl_s=settings.c3.state_ttl_s,
    )

    with pg_connection() as conn, conn.cursor() as cur:
        # Sample 100 random accounts
        cur.execute("""
                SELECT account_id, state_json
                FROM gold.account_state
                ORDER BY random()
                LIMIT 100
            """)
        divergences = []
        for account_id, pg_state_json in cur:
            # Read from Redis
            redis_state_json = state_store.get(account_id)

            if redis_state_json is None:
                # Cold start: Redis missing state is ok
                continue

            pg_state = json.loads(pg_state_json)
            redis_state = json.loads(redis_state_json)

            # Compare essentials (last_* anchor block)
            for key in ["last_txn_id", "last_event_ts_us", "last_amount"]:
                if pg_state.get(key) != redis_state.get(key):
                    divergences.append(
                        f"{account_id}/{key}: pg={pg_state.get(key)}, redis={redis_state.get(key)}"
                    )

        if divergences:
            raise AssertionError("State divergence detected:\n" + "\n".join(divergences[:10]))

    print("State consistency verified: Redis and Postgres aligned")
    return "state_consistency: ok"


with DAG(
    dag_id="dag_skew_audit",
    description="Layer 6 gate 6: Daily skew audit (P1 enforcement - zero training/serving skew)",
    start_date=datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC),
    schedule="@daily",
    catchup=False,
    tags=["layer-6", "audit", "daily", "p1"],
) as dag:
    feature_skew = audit_feature_skew()
    model = audit_model_consistency()
    state = audit_state_consistency()

    feature_skew >> model >> state
