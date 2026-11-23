"""DuckDB implementation of the Storage port: one local file, no server, all machines in it."""

import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType

import duckdb
import pyarrow as pa

from ..contracts import Contract, FailureEvent, Measurement, Quality

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS measurements (
        machine_id VARCHAR NOT NULL,
        sensor_id VARCHAR NOT NULL,
        ts TIMESTAMPTZ NOT NULL,
        value DOUBLE NOT NULL,
        quality VARCHAR NOT NULL DEFAULT 'good'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS failure_events (
        machine_id VARCHAR NOT NULL,
        ts TIMESTAMPTZ NOT NULL,
        failure_mode VARCHAR NOT NULL,
        description VARCHAR NOT NULL DEFAULT ''
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS results (
        id VARCHAR PRIMARY KEY,
        kind VARCHAR NOT NULL,
        created_at TIMESTAMPTZ NOT NULL,
        payload JSON NOT NULL
    )
    """,
)

_MEASUREMENT_COLUMNS = ("machine_id", "sensor_id", "ts", "value", "quality")
_MEASUREMENT_KEY = ("machine_id", "sensor_id", "ts")
_EVENT_COLUMNS = ("machine_id", "ts", "failure_mode", "description")
_EVENT_KEY = ("machine_id", "ts", "failure_mode")
_TS = pa.timestamp("us", tz="UTC")


@dataclass(frozen=True)
class SensorSummary:
    sensor_id: str
    readings: int
    first: datetime
    last: datetime
    minimum: float
    maximum: float
    mean: float


def _naive_utc(value: datetime) -> datetime:
    return value.astimezone(UTC).replace(tzinfo=None)


def _utc(value: datetime) -> datetime:
    """Attach UTC to a naive datetime produced by `AT TIME ZONE 'UTC'`."""
    return value.replace(tzinfo=UTC)


class DuckDBStorage:
    def __init__(self, path: Path | str) -> None:
        location = str(path)
        if location != ":memory:":
            Path(location).parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(location)
        self._con.execute("SET TimeZone='UTC'")
        for statement in _SCHEMA:
            self._con.execute(statement)

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> "DuckDBStorage":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # --- measurements -------------------------------------------------------------------------

    def write_measurements(self, rows: Sequence[Measurement]) -> int:
        if not rows:
            return 0
        table = pa.table(
            {
                "machine_id": pa.array([r.machine_id for r in rows], pa.string()),
                "sensor_id": pa.array([r.sensor_id for r in rows], pa.string()),
                "ts": pa.array([r.timestamp for r in rows], _TS),
                "value": pa.array([r.value for r in rows], pa.float64()),
                "quality": pa.array([r.quality.value for r in rows], pa.string()),
            }
        )
        return self.write_table(table)

    def write_table(self, table: pa.Table) -> int:
        """Bulk insert of columns machine_id, sensor_id, ts, value, quality. Skips known rows."""
        return self._insert_new("measurements", _MEASUREMENT_COLUMNS, _MEASUREMENT_KEY, table)

    def read_timeseries(
        self, machine_id: str, sensor_id: str, start: datetime, end: datetime
    ) -> list[Measurement]:
        rows = self._con.execute(
            """
            SELECT ts AT TIME ZONE 'UTC', value, quality
            FROM measurements
            WHERE machine_id = ? AND sensor_id = ?
              AND ts >= (CAST(? AS TIMESTAMP) AT TIME ZONE 'UTC')
              AND ts < (CAST(? AS TIMESTAMP) AT TIME ZONE 'UTC')
            ORDER BY ts
            """,
            [machine_id, sensor_id, _naive_utc(start), _naive_utc(end)],
        ).fetchall()
        return [
            Measurement(
                timestamp=_utc(ts),
                machine_id=machine_id,
                sensor_id=sensor_id,
                value=value,
                quality=Quality(quality),
            )
            for ts, value, quality in rows
        ]

    def read_table(
        self, machine_id: str, start: datetime | None = None, end: datetime | None = None
    ) -> pa.Table:
        """Bulk read for analysis: columns sensor_id, ts, value, ordered by time.

        Readings marked 'bad' are left out. Optional range: start <= ts < end.
        """
        lo = _naive_utc(start) if start else datetime(1, 1, 1)
        hi = _naive_utc(end) if end else datetime(9999, 12, 31)
        return self._con.sql(
            """
            SELECT sensor_id, ts, value FROM measurements
            WHERE machine_id = $m AND quality <> 'bad'
              AND ts >= (CAST($lo AS TIMESTAMP) AT TIME ZONE 'UTC')
              AND ts < (CAST($hi AS TIMESTAMP) AT TIME ZONE 'UTC')
            ORDER BY ts, sensor_id
            """,
            params={"m": machine_id, "lo": lo, "hi": hi},
        ).to_arrow_table()

    def summary(self, machine_id: str) -> list[SensorSummary]:
        rows = self._con.execute(
            """
            SELECT sensor_id, count(*), min(ts AT TIME ZONE 'UTC'), max(ts AT TIME ZONE 'UTC'),
                   min(value), max(value), avg(value)
            FROM measurements WHERE machine_id = ?
            GROUP BY sensor_id ORDER BY sensor_id
            """,
            [machine_id],
        ).fetchall()
        return [
            SensorSummary(sid, count, _utc(first), _utc(last), low, high, mean)
            for sid, count, first, last, low, high, mean in rows
        ]

    # --- failure events -----------------------------------------------------------------------

    def write_events(self, events: Sequence[FailureEvent]) -> int:
        if not events:
            return 0
        table = pa.table(
            {
                "machine_id": pa.array([e.machine_id for e in events], pa.string()),
                "ts": pa.array([e.timestamp for e in events], _TS),
                "failure_mode": pa.array([e.failure_mode for e in events], pa.string()),
                "description": pa.array([e.description for e in events], pa.string()),
            }
        )
        return self.write_events_table(table)

    def write_events_table(self, table: pa.Table) -> int:
        """Bulk insert of columns machine_id, ts, failure_mode, description. Skips known rows."""
        return self._insert_new("failure_events", _EVENT_COLUMNS, _EVENT_KEY, table)

    def read_events(self, machine_id: str) -> list[FailureEvent]:
        rows = self._con.execute(
            """
            SELECT ts AT TIME ZONE 'UTC', failure_mode, description
            FROM failure_events WHERE machine_id = ? ORDER BY ts
            """,
            [machine_id],
        ).fetchall()
        return [
            FailureEvent(
                machine_id=machine_id, timestamp=_utc(ts), failure_mode=mode, description=desc
            )
            for ts, mode, desc in rows
        ]

    # --- results ------------------------------------------------------------------------------

    def write_result(self, kind: str, payload: Contract) -> str:
        result_id = uuid.uuid4().hex[:12]
        self._con.execute(
            "INSERT INTO results VALUES (?, ?, ?, ?)",
            [result_id, kind, _naive_utc(datetime.now(UTC)), payload.model_dump_json()],
        )
        return f"{kind}:{result_id}"

    def read_result(self, ref: str) -> dict:
        kind, _, result_id = ref.partition(":")
        row = self._con.execute(
            "SELECT payload FROM results WHERE kind = ? AND id = ?", [kind, result_id]
        ).fetchone()
        if row is None:
            raise KeyError(f"result {ref!r} not found")
        return json.loads(row[0])

    # --- internals ----------------------------------------------------------------------------

    def _insert_new(
        self, target: str, columns: Sequence[str], key: Sequence[str], table: pa.Table
    ) -> int:
        """Insert rows whose key is not stored yet (also collapses duplicates inside the batch).

        `target`, `columns` and `key` are internal constants, never user input.
        """
        cols = ", ".join(columns)
        selected = ", ".join(f"i.{c}" for c in columns)
        partition = ", ".join(f"i.{k}" for k in key)
        match = " AND ".join(f"m.{k} = i.{k}" for k in key)
        order = ", ".join(key)
        self._con.register("incoming", table)
        try:
            rows = self._con.execute(
                f"""
                INSERT INTO {target} ({cols})
                SELECT {cols} FROM (
                    SELECT {selected}, row_number() OVER (PARTITION BY {partition}) AS rn
                    FROM incoming i
                    WHERE NOT EXISTS (SELECT 1 FROM {target} m WHERE {match})
                ) WHERE rn = 1
                ORDER BY {order}
                """
            ).fetchall()
        finally:
            self._con.unregister("incoming")
        return int(rows[0][0]) if rows else 0
