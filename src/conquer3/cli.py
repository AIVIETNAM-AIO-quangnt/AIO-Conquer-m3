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

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = args.handler
    return int(handler(args))


if __name__ == "__main__":
    sys.exit(main())
