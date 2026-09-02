"""Console entry point.

Every subcommand imports its dependencies **lazily, inside the handler**. A
top-level ``import duckdb`` here would make ``conquer3 --help`` fail in the serving
image, which deliberately has no pipeline extras installed.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conquer3.contracts.model_registry import ModelRef


def _cmd_version(_: argparse.Namespace) -> int:
    from conquer3 import __version__
    from conquer3.core.schema import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, STATE_SCHEMA_VERSION

    print(f"conquer3 {__version__}")
    print(f"feature_schema_version={FEATURE_SCHEMA_VERSION}")
    print(f"state_schema_version={STATE_SCHEMA_VERSION}")
    print(f"n_features={len(FEATURE_NAMES)}")
    return 0


def _cmd_features_list(_: argparse.Namespace) -> int:
    from conquer3.core.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES

    for name in NUMERIC_FEATURES:
        print(f"numeric\t{name}")
    for name in CATEGORICAL_FEATURES:
        print(f"categorical\t{name}")
    return 0


def _cmd_db_migrate(_: argparse.Namespace) -> int:
    from conquer3.db.bootstrap import apply_ddl
    from conquer3.db.engine import pg_connection

    with pg_connection() as conn:
        applied = apply_ddl(conn)
    for name in applied:
        print(f"applied\t{name}")
    return 0


def _cmd_db_gen_gold_ddl(args: argparse.Namespace) -> int:
    from conquer3.db.ddl_gen import GOLD_DDL_PATH, render_gold_ddl, write_gold_ddl

    rendered = render_gold_ddl()
    if args.check:
        current = GOLD_DDL_PATH.read_text() if GOLD_DDL_PATH.is_file() else ""
        if current != rendered:
            msg = f"{GOLD_DDL_PATH} is out of date; run `conquer3 db gen-gold-ddl`"
            print(msg, file=sys.stderr)
            return 1
        print(f"{GOLD_DDL_PATH} is up to date")
        return 0
    write_gold_ddl()
    print(f"wrote {GOLD_DDL_PATH}")
    return 0


def _cmd_ingest_download(args: argparse.Namespace) -> int:
    from conquer3.config.settings import get_settings
    from conquer3.pipelines.ingest.kaggle import download_paysim_csv

    dest = args.dest or get_settings().kaggle.csv_path
    path = download_paysim_csv(dest)
    print(f"downloaded\t{path}")
    return 0


def _cmd_ingest_bronze(args: argparse.Namespace) -> int:
    from conquer3.config.settings import get_settings
    from conquer3.pipelines.ingest.bronze import load_csv_to_bronze

    csv_path = args.csv or get_settings().kaggle.csv_path
    row_count = load_csv_to_bronze(csv_path)
    print(f"bronze.txn_raw\t{row_count} rows")
    return 0


def _cmd_transform_bronze_to_silver(_: argparse.Namespace) -> int:
    from conquer3.pipelines.transforms.bronze_to_silver import bronze_to_silver

    row_count = bronze_to_silver()
    print(f"silver.txn\t{row_count} rows")
    return 0


def _cmd_transform_silver_to_gold(_: argparse.Namespace) -> int:
    from conquer3.pipelines.transforms.silver_to_gold import silver_to_gold

    row_count = silver_to_gold()
    print(f"gold.txn_features\t{row_count} rows")
    return 0


def _cmd_transform_export_staging(_: argparse.Namespace) -> int:
    from conquer3.pipelines.transforms.export_staging import export_staging

    row_count = export_staging()
    print(f"staging/ctx\t{row_count} rows")
    return 0


def _cmd_pathway_backfill(_: argparse.Namespace) -> int:
    from conquer3.pipelines.pathway.run_backfill import main as run_backfill_main

    return run_backfill_main()


def _cmd_pathway_streaming(_: argparse.Namespace) -> int:
    from conquer3.pipelines.pathway.run_streaming import main as run_streaming_main

    return run_streaming_main()


def _cmd_model_publish_dummy(args: argparse.Namespace) -> int:
    import subprocess

    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn.dummy import DummyClassifier

    from conquer3.contracts.model_registry import publish_model
    from conquer3.core.schema import CATEGORICAL_FEATURES, FEATURE_NAMES, NUMERIC_FEATURES

    rng = np.random.default_rng(0)
    n = 20
    data: dict[str, object] = {name: rng.normal(size=n) for name in NUMERIC_FEATURES}
    for name in CATEGORICAL_FEATURES:
        data[name] = rng.choice(["a", "b"], size=n)
    x_sample = pd.DataFrame(data, columns=list(FEATURE_NAMES))
    y = rng.integers(0, 2, size=n)

    clf = DummyClassifier(strategy="prior").fit(x_sample, y)
    proba = clf.predict_proba(x_sample)

    try:
        code_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        code_sha = "unknown"

    ref = publish_model(
        clf,
        x_sample,
        proba,
        sklearn_version=sklearn.__version__,
        code_sha=code_sha,
        decision_threshold=0.5,
        model_name=args.name,
        alias_as_champion=args.alias_champion,
    )
    print(f"published\t{ref.name}\tversion={ref.version}\trun_id={ref.run_id}")
    return 0


def _cmd_model_resolve_champion(args: argparse.Namespace) -> int:
    from conquer3.contracts.model_registry import resolve_champion

    _model, ref = resolve_champion(args.name)
    print(f"resolved\t{ref.name}\tversion={ref.version}\tdegraded={ref.degraded}")
    return 0


def _record_deployment(ref: ModelRef) -> None:
    # Lives here, not in conquer3.serving, because import-linter forbids
    # conquer3.serving from ever importing conquer3.db (see db/ops.py's
    # record_model_deployment docstring). A failed audit-trail write must never
    # take the scorer down -- Postgres being unavailable is not the property the
    # Layer 5 gate defends; a dead remote MLflow is.
    import logging

    from conquer3.db.engine import pg_connection
    from conquer3.db.ops import record_model_deployment

    try:
        with pg_connection() as conn:
            record_model_deployment(conn, ref)
    except Exception:
        logging.getLogger(__name__).warning(
            "failed to record model deployment for version %s", ref.version, exc_info=True
        )


def _cmd_replay(args: argparse.Namespace) -> int:
    from conquer3.config.settings import get_settings
    from conquer3.producer.replay import run_replay

    settings = get_settings()
    csv_path = args.csv or settings.kaggle.csv_path
    endpoint = args.endpoint or f"http://127.0.0.1:{settings.serving.scorer_port}"
    run_replay(
        csv_path,
        args.out,
        endpoint=endpoint,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        timeout_s=args.timeout,
        limit=args.limit,
    )
    return 0


def _cmd_serve(_: argparse.Namespace) -> int:
    from conquer3.config.settings import get_settings
    from conquer3.serving.supervisor import serve
    from conquer3.telemetry.otel import init_telemetry

    init_telemetry("conquer3-scorer")
    return serve(get_settings(), on_deployment=_record_deployment)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="conquer3", description="Credit-fraud MLOps platform")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="print version and schema versions").set_defaults(
        handler=_cmd_version
    )

    features = sub.add_parser("features", help="feature schema utilities")
    features_sub = features.add_subparsers(dest="features_command", required=True)
    features_sub.add_parser("list", help="list features and their types").set_defaults(
        handler=_cmd_features_list
    )

    db = sub.add_parser("db", help="warehouse schema utilities")
    db_sub = db.add_subparsers(dest="db_command", required=True)
    db_sub.add_parser("migrate", help="apply db/ddl/*.sql idempotently").set_defaults(
        handler=_cmd_db_migrate
    )
    gen_gold = db_sub.add_parser(
        "gen-gold-ddl", help="regenerate db/ddl/30_gold.sql from core.schema"
    )
    gen_gold.add_argument(
        "--check", action="store_true", help="fail if the committed file is out of date"
    )
    gen_gold.set_defaults(handler=_cmd_db_gen_gold_ddl)

    ingest = sub.add_parser("ingest", help="land raw data into the bronze layer")
    ingest_sub = ingest.add_subparsers(dest="ingest_command", required=True)
    download = ingest_sub.add_parser("download", help="download the PaySim1 CSV from Kaggle")
    dest_help = "output CSV path (default: C3_PAYSIM_CSV_PATH from .env)"
    download.add_argument("--dest", default=None, help=dest_help)
    download.set_defaults(handler=_cmd_ingest_download)
    bronze = ingest_sub.add_parser("bronze", help="load a PaySim1 CSV into bronze.txn_raw")
    csv_help = "input CSV path (default: C3_PAYSIM_CSV_PATH from .env)"
    bronze.add_argument("--csv", default=None, help=csv_help)
    bronze.set_defaults(handler=_cmd_ingest_bronze)

    transform = sub.add_parser("transform", help="medallion transforms")
    transform_sub = transform.add_subparsers(dest="transform_command", required=True)
    transform_sub.add_parser(
        "bronze-to-silver", help="type/clean bronze.txn_raw into silver.txn"
    ).set_defaults(handler=_cmd_transform_bronze_to_silver)
    transform_sub.add_parser(
        "silver-to-gold", help="compute features from silver.txn into gold.txn_features"
    ).set_defaults(handler=_cmd_transform_silver_to_gold)
    transform_sub.add_parser(
        "export-staging", help="export silver.txn to JSONL staging for Pathway"
    ).set_defaults(handler=_cmd_transform_export_staging)

    pathway = sub.add_parser("pathway", help="Pathway feature engine (Layer 3b)")
    pathway_sub = pathway.add_subparsers(dest="pathway_command", required=True)
    pathway_sub.add_parser(
        "backfill", help="static-mode: fold the staging snapshot once into account state"
    ).set_defaults(handler=_cmd_pathway_backfill)
    pathway_sub.add_parser(
        "streaming", help="streaming-mode: continuously repair account state"
    ).set_defaults(handler=_cmd_pathway_streaming)

    model = sub.add_parser("model", help="MLflow model registry contract (Layer 4)")
    model_sub = model.add_subparsers(dest="model_command", required=True)
    publish_dummy = model_sub.add_parser(
        "publish-dummy", help="publish a DummyClassifier -- smoke-tests the registry contract"
    )
    publish_dummy.add_argument("--name", default=None, help="model name (default: C3_MODEL_NAME)")
    publish_dummy.add_argument(
        "--alias-champion", action="store_true", help='also alias the new version "champion"'
    )
    publish_dummy.set_defaults(handler=_cmd_model_publish_dummy)
    resolve_champion = model_sub.add_parser(
        "resolve-champion", help='resolve the "champion" alias (live, falling back to cache)'
    )
    resolve_champion.add_argument(
        "--name", default=None, help="model name (default: C3_MODEL_NAME)"
    )
    resolve_champion.set_defaults(handler=_cmd_model_resolve_champion)

    sub.add_parser(
        "serve", help="run the scoring service: resolve champion, serve /predict (Layer 5)"
    ).set_defaults(handler=_cmd_serve)

    replay = sub.add_parser(
        "replay", help="replay a raw PaySim1 CSV against /predict, for offline evaluation"
    )
    replay.add_argument("--csv", default=None, help="input CSV (default: C3_PAYSIM_CSV_PATH)")
    replay.add_argument(
        "--out", required=True, help="output CSV: ground truth + prediction, one row each"
    )
    replay.add_argument(
        "--endpoint",
        default=None,
        help="scorer base URL (default: http://127.0.0.1:$C3_SCORER_PORT)",
    )
    replay.add_argument(
        "--batch-size", type=int, default=200, help="transactions per /predict call"
    )
    replay.add_argument("--limit", type=int, default=None, help="only replay the first N rows")
    replay.add_argument(
        "--dry-run", action="store_true", help="score without writing Redis state or event logs"
    )
    replay.add_argument("--timeout", type=float, default=30.0, help="per-request timeout, seconds")
    replay.set_defaults(handler=_cmd_replay)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = args.handler
    return int(handler(args))


if __name__ == "__main__":
    sys.exit(main())
