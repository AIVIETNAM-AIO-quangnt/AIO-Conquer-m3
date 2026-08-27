-- Layer 2: the medallion warehouse. Applied via docker-entrypoint-initdb.d on a
-- fresh volume, and idempotently at any time via `conquer3 db migrate`
-- (conquer3.db.bootstrap.apply_ddl) -- initdb only ever runs once per volume, so an
-- already-initialized Layer 1 volume needs the explicit path.
CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;
CREATE SCHEMA IF NOT EXISTS ops;
