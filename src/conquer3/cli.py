"""Console entry point.

Every subcommand imports its dependencies **lazily, inside the handler**. A
top-level ``import duckdb`` here would make ``conquer3 --help`` fail in the serving
image, which deliberately has no pipeline extras installed.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


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

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = args.handler
    return int(handler(args))


if __name__ == "__main__":
    sys.exit(main())
