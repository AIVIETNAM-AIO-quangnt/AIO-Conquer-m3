"""BentoML scoring service.

Must not import `conquer3.db`, `conquer3.pipelines`, duckdb, ibis, pathway or polars:
the serving image installs none of them. Enforced by import-linter.
"""
