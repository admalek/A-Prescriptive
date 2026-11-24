"""File ingestion (interface I1): CSV or Parquet files into the storage, for any machine.

Two layouts are understood:
  wide  one timestamp column and one numeric column per sensor (typical export from a PLC/SCADA)
  long  one row per reading: timestamp, sensor name, value (typical export from a historian)

Timestamps with an explicit offset ("...Z", "+01:00") are taken as they are; timestamps without
one are interpreted in the time zone given by `tz`. Everything is stored in UTC. Importing the same
file twice stores nothing twice.
"""

import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pyarrow as pa

from ..machines import validate_machine_id
from ..storage import DuckDBStorage

_CSV_SUFFIXES = {".csv", ".tsv", ".txt"}
_NUMERIC = {
    "TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT",
    "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT", "UHUGEINT",
    "FLOAT", "DOUBLE",
}  # fmt: skip
_TZ = re.compile(r"^[A-Za-z0-9_+\-/]{1,64}$")


class IngestError(Exception):
    """The file cannot be imported as requested; the message says why."""


@dataclass(frozen=True)
class IngestReport:
    file: str
    machine_id: str
    layout: str
    rows_in_file: int
    readings_found: int  # readings with a valid timestamp, sensor and number
    readings_rejected: int  # empty or invalid cells
    readings_new: int
    readings_duplicate: int  # already stored, skipped
    sensors: list[str]
    start: datetime | None
    end: datetime | None
    ignored_columns: list[str]


@dataclass(frozen=True)
class EventsReport:
    file: str
    machine_id: str
    rows_in_file: int
    events_found: int
    events_new: int
    events_duplicate: int


def slugify(name: str) -> str:
    """'Temperatura łożyska [°C]' -> 'temperatura_lozyska_c'."""
    text = name.replace("ł", "l").replace("Ł", "L")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def ingest_file(
    storage: DuckDBStorage,
    machine_id: str,
    path: Path,
    *,
    layout: str = "wide",
    timestamp_col: str = "timestamp",
    sensor_col: str = "sensor_id",
    value_col: str = "value",
    tz: str = "UTC",
    timestamp_format: str | None = None,
    decimal: str = ".",
) -> IngestReport:
    validate_machine_id(machine_id)
    if layout not in {"wide", "long"}:
        raise IngestError(f"unknown layout {layout!r}: use 'wide' or 'long'")
    con, rel, ts_col = _open(path, timestamp_col, timestamp_format, decimal, tz)
    types = {name: str(t) for name, t in zip(rel.columns, rel.types, strict=True)}
    ts_expr = _ts_expr(ts_col, types[ts_col], timestamp_format)
    ignored: list[str] = []

    try:
        if layout == "wide":
            pairs = []  # (column in file, sensor id)
            for name in rel.columns:
                if name == ts_col:
                    continue
                if _is_numeric(types[name]):
                    pairs.append((name, slugify(name)))
                else:
                    ignored.append(name)
            if not pairs:
                raise IngestError(f"no numeric sensor columns found in {path.name}")
            slugs = [slug for _, slug in pairs]
            if "" in slugs or len(set(slugs)) != len(slugs):
                raise IngestError("sensor column names are empty or collide after normalisation")
            casts = ", ".join(f"CAST({_q(c)} AS DOUBLE) AS {_q(s)}" for c, s in pairs)
            in_list = ", ".join(_q(s) for s in slugs)
            con.execute(
                f"""
                CREATE TEMP VIEW long_raw AS
                WITH base AS (SELECT {ts_expr} AS ts, {casts} FROM src)
                SELECT ts, sensor_id, value
                FROM base UNPIVOT INCLUDE NULLS (value FOR sensor_id IN ({in_list}))
                """
            )
        else:
            sensor = _resolve(rel.columns, sensor_col, "sensor")
            value = _resolve(rel.columns, value_col, "value")
            raw_names = [
                row[0]
                for row in con.execute(
                    f"SELECT DISTINCT CAST({_q(sensor)} AS VARCHAR) FROM src "
                    f"WHERE {_q(sensor)} IS NOT NULL"
                ).fetchall()
            ]
            slug_of = {raw: slugify(raw) for raw in raw_names}
            collisions = [s for s in set(slug_of.values()) if list(slug_of.values()).count(s) > 1]
            if collisions:
                raise IngestError(f"sensor names collide after normalisation: {collisions}")
            con.register(
                "sensor_map",
                pa.table(
                    {
                        "raw": pa.array(list(slug_of), pa.string()),
                        "slug": pa.array([s or None for s in slug_of.values()], pa.string()),
                    }
                ),
            )
            con.execute(
                f"""
                CREATE TEMP VIEW long_raw AS
                SELECT {ts_expr} AS ts, m.slug AS sensor_id,
                       TRY_CAST(s.{_q(value)} AS DOUBLE) AS value
                FROM src s LEFT JOIN sensor_map m ON CAST(s.{_q(sensor)} AS VARCHAR) = m.raw
                """
            )

        good = "ts IS NOT NULL AND sensor_id IS NOT NULL AND value IS NOT NULL AND isfinite(value)"
        rows_in_file = _scalar(con, "SELECT count(*) FROM src")
        total, found, first, last = _rows(
            con,
            f"""
            SELECT count(*), count(*) FILTER (WHERE {good}),
                   min(ts AT TIME ZONE 'UTC') FILTER (WHERE {good}),
                   max(ts AT TIME ZONE 'UTC') FILTER (WHERE {good})
            FROM long_raw
            """,
        )[0]
        sensors = [
            row[0]
            for row in _rows(
                con, f"SELECT DISTINCT sensor_id FROM long_raw WHERE {good} ORDER BY sensor_id"
            )
        ]
        table = _arrow(
            con.sql(
                f"""
                SELECT '{machine_id}' AS machine_id, sensor_id, ts, value, 'good' AS quality
                FROM long_raw WHERE {good} ORDER BY sensor_id, ts
                """
            )
        )
    except duckdb.Error as exc:
        raise IngestError(f"cannot import {path.name}: {exc}") from exc
    finally:
        con.close()

    new = storage.write_table(table) if found else 0
    return IngestReport(
        file=path.name,
        machine_id=machine_id,
        layout=layout,
        rows_in_file=rows_in_file,
        readings_found=found,
        readings_rejected=total - found,
        readings_new=new,
        readings_duplicate=found - new,
        sensors=sensors,
        start=first.replace(tzinfo=UTC) if first else None,
        end=last.replace(tzinfo=UTC) if last else None,
        ignored_columns=ignored,
    )


def ingest_events(
    storage: DuckDBStorage,
    machine_id: str,
    path: Path,
    *,
    timestamp_col: str = "timestamp",
    mode_col: str = "failure_mode",
    description_col: str = "description",
    tz: str = "UTC",
    timestamp_format: str | None = None,
    decimal: str = ".",
) -> EventsReport:
    """Import labelled failures (timestamp, failure mode, optional description)."""
    validate_machine_id(machine_id)
    con, rel, ts_col = _open(path, timestamp_col, timestamp_format, decimal, tz)
    types = {name: str(t) for name, t in zip(rel.columns, rel.types, strict=True)}
    ts_expr = _ts_expr(ts_col, types[ts_col], timestamp_format)
    mode = _resolve(rel.columns, mode_col, "failure mode")
    description = _resolve(rel.columns, description_col, "description", required=False)
    description_expr = f"coalesce(CAST({_q(description)} AS VARCHAR), '')" if description else "''"
    good = "ts IS NOT NULL AND failure_mode IS NOT NULL AND failure_mode <> ''"
    try:
        con.execute(
            f"""
            CREATE TEMP VIEW events_raw AS
            SELECT '{machine_id}' AS machine_id, {ts_expr} AS ts,
                   trim(CAST({_q(mode)} AS VARCHAR)) AS failure_mode,
                   {description_expr} AS description
            FROM src
            """
        )
        rows_in_file = _scalar(con, "SELECT count(*) FROM src")
        table = _arrow(con.sql(f"SELECT * FROM events_raw WHERE {good} ORDER BY ts"))
    except duckdb.Error as exc:
        raise IngestError(f"cannot import {path.name}: {exc}") from exc
    finally:
        con.close()

    found = table.num_rows
    new = storage.write_events_table(table) if found else 0
    return EventsReport(path.name, machine_id, rows_in_file, found, new, found - new)


# --- helpers ----------------------------------------------------------------------------------


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _is_numeric(type_name: str) -> bool:
    return type_name in _NUMERIC or type_name.startswith("DECIMAL")


def _rows(con: duckdb.DuckDBPyConnection, sql: str) -> list[tuple]:
    """Run a query and read ALL rows, so the connection is free for the next statement
    (a partly read result makes DuckDB refuse further work on the same connection)."""
    return con.execute(sql).fetchall()


def _scalar(con: duckdb.DuckDBPyConnection, sql: str):
    return _rows(con, sql)[0][0]


def _arrow(relation) -> pa.Table:
    fetch = getattr(relation, "to_arrow_table", None) or relation.fetch_arrow_table
    return fetch()


def _resolve(columns: list[str], wanted: str, what: str, *, required: bool = True) -> str | None:
    if wanted in columns:
        return wanted
    matches = [c for c in columns if c.lower() == wanted.lower()]
    if len(matches) == 1:
        return matches[0]
    if not required:
        return None
    raise IngestError(
        f"{what} column {wanted!r} not found; columns in the file: {', '.join(columns)}"
    )


def _ts_expr(column: str, type_name: str, timestamp_format: str | None) -> str:
    """SQL turning the timestamp column into TIMESTAMPTZ, using the connection's time zone
    for values that carry no offset."""
    col = _q(column)
    if type_name == "VARCHAR":
        if timestamp_format:
            fmt = timestamp_format.replace("'", "''")
            return f"CAST(try_strptime({col}, '{fmt}') AS TIMESTAMPTZ)"
        return f"TRY_CAST({col} AS TIMESTAMPTZ)"
    if type_name.startswith("TIMESTAMP") or type_name == "DATE":
        return f"CAST({col} AS TIMESTAMPTZ)"
    raise IngestError(f"column {column!r} has type {type_name}; expected timestamps")


def _open(
    path: Path, timestamp_col: str, timestamp_format: str | None, decimal: str, tz: str
) -> tuple[duckdb.DuckDBPyConnection, duckdb.DuckDBPyRelation, str]:
    if not path.is_file():
        raise IngestError(f"file not found: {path}")
    if not _TZ.fullmatch(tz):
        raise IngestError(f"invalid time zone {tz!r}; use a name like Europe/Warsaw or UTC")
    suffix = path.suffix.lower()
    if suffix not in _CSV_SUFFIXES and suffix != ".parquet":
        raise IngestError(f"unsupported file type {suffix!r}: use .csv, .tsv, .txt or .parquet")

    con = duckdb.connect(":memory:")
    try:
        con.execute(f"SET TimeZone='{tz}'")
        if suffix == ".parquet":
            raw = con.read_parquet(str(path))
            ts_col = _resolve(raw.columns, timestamp_col, "timestamp")
        else:
            options = {"decimal": decimal} if decimal != "." else {}
            raw = con.read_csv(str(path), **options)
            ts_col = _resolve(raw.columns, timestamp_col, "timestamp")
            # Parse timestamps ourselves from text, so offsets and the time zone are applied
            # consistently instead of being guessed by the CSV reader.
            if str(raw.types[raw.columns.index(ts_col)]) != "VARCHAR":
                raw = con.read_csv(str(path), dtype={ts_col: "VARCHAR"}, **options)
        # Load into a temporary table first. Casting directly on top of the CSV scan would be
        # pushed into the reader, where one bad cell is a hard error instead of a NULL
        # (= a rejected row that we count and report).
        raw.to_table("src")
        return con, con.table("src"), ts_col
    except duckdb.Error as exc:
        con.close()
        raise IngestError(f"cannot read {path.name}: {exc}") from exc
    except IngestError:
        con.close()
        raise
