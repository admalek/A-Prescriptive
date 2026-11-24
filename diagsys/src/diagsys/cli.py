"""Command line interface: doctor, machine, simulate, ingest, ingest-events, data summary,
quality, clean, features, import-xlsx, xlsx-template."""

from pathlib import Path

import numpy as np
import typer
import yaml

from . import __version__
from .config import load_settings
from .contracts import QualityReport
from .data import (
    FLAG_IMPUTED,
    FeatureError,
    FrameError,
    IngestError,
    XlsxError,
    import_workbook,
    ingest_events,
    ingest_file,
    prepare,
    write_outputs,
    write_template,
)
from .doctor import run_checks
from .gateway import SimConfig, write_simulation
from .gateway import simulate as run_simulation
from .machines import (
    MachineError,
    add_sensors,
    create_machine,
    list_machines,
    load_machine,
    machine_dir,
)
from .storage import DuckDBStorage

app = typer.Typer(
    help="Universal local multi-agent system for machine diagnostics.",
    no_args_is_help=True,
    add_completion=False,
)
machine_app = typer.Typer(help="Machine profiles.", no_args_is_help=True)
data_app = typer.Typer(help="Inspect stored data.", no_args_is_help=True)
app.add_typer(machine_app, name="machine")
app.add_typer(data_app, name="data")

_LABEL = {"ok": "[OK]  ", "warn": "[WARN]", "fail": "[FAIL]"}


def _fail(exc: Exception) -> typer.Exit:
    typer.echo(f"Error: {exc}", err=True)
    return typer.Exit(code=1)


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(__version__)


@app.command()
def doctor() -> None:
    """Check whether this computer is ready (RAM, disk, model server, packages)."""
    checks = run_checks(load_settings())
    for check in checks:
        typer.echo(f"{_LABEL[check.status]} {check.name}: {check.detail}")
    if any(check.status == "fail" for check in checks):
        raise typer.Exit(code=1)


# --- machines ---------------------------------------------------------------------------------


@machine_app.command("new")
def machine_new(
    machine_id: str = typer.Argument(..., help="Short id, e.g. press-01"),
    name: str = typer.Option(None, "--name", "-n", help="Display name (default: the id)"),
    machine_type: str = typer.Option("generic", "--type", "-t", help="Machine type"),
) -> None:
    """Create a new machine profile with its own docs, data, models and reports folders."""
    settings = load_settings()
    try:
        profile = create_machine(settings, machine_id, name or machine_id, machine_type)
    except MachineError as exc:
        raise _fail(exc) from exc
    typer.echo(f"Created machine {profile.machine_id} in {settings.machines_dir / machine_id}")
    typer.echo(f"Put its documentation in docs/; RAG collection: {profile.rag_collection}")


@machine_app.command("list")
def machine_list() -> None:
    """List machine profiles."""
    profiles = list_machines(load_settings())
    if not profiles:
        typer.echo("No machines yet. Create one with: diagsys machine new <id>")
        return
    for profile in profiles:
        typer.echo(f"{profile.machine_id}\t{profile.machine_type}\t{profile.name}")


@machine_app.command("show")
def machine_show(machine_id: str = typer.Argument(...)) -> None:
    """Show one machine profile."""
    try:
        profile = load_machine(load_settings(), machine_id)
    except MachineError as exc:
        raise _fail(exc) from exc
    typer.echo(yaml.safe_dump(profile.model_dump(mode="json"), sort_keys=False, allow_unicode=True))


# --- data -------------------------------------------------------------------------------------


@app.command()
def simulate(
    machine_id: str = typer.Argument(..., help="An existing machine (diagsys machine new)"),
    cycles: int = typer.Option(5, min=1, help="Run-to-failure cycles to generate"),
    sampling_s: int = typer.Option(60, min=1, help="Seconds between readings"),
    seed: int = typer.Option(42, help="Same seed gives the same data"),
    dirty: bool = typer.Option(False, "--dirty", help="Add missing values, spikes, a stuck sensor"),
) -> None:
    """Generate synthetic run-to-failure data as files in the machine's raw/ folder."""
    settings = load_settings()
    try:
        load_machine(settings, machine_id)
    except MachineError as exc:
        raise _fail(exc) from exc
    result = run_simulation(
        SimConfig(cycles=cycles, sampling_s=sampling_s, seed=seed, dirty=dirty), machine_id
    )
    raw = machine_dir(settings, machine_id) / "raw"
    csv_path = raw / f"sim_seed{seed}.csv"
    events_path = raw / f"sim_seed{seed}_failures.csv"
    write_simulation(result, csv_path, events_path)
    typer.echo(f"Wrote {result.timestamps.size} rows x {len(result.values)} sensors: {csv_path}")
    typer.echo(f"Wrote {len(result.failures)} failure labels: {events_path}")
    if dirty:
        typer.echo(f"Injected: {result.injected}")
    typer.echo("Next:")
    typer.echo(f"  diagsys ingest {machine_id} {csv_path}")
    typer.echo(f"  diagsys ingest-events {machine_id} {events_path}")


@app.command()
def ingest(
    machine_id: str = typer.Argument(...),
    file: Path = typer.Argument(..., help="CSV, TSV, TXT or Parquet file"),
    layout: str = typer.Option(
        "wide", help="wide: a column per sensor; long: timestamp/sensor/value"
    ),
    timestamp_col: str = typer.Option("timestamp", help="Name of the timestamp column"),
    sensor_col: str = typer.Option("sensor_id", help="Long layout: sensor name column"),
    value_col: str = typer.Option("value", help="Long layout: value column"),
    tz: str = typer.Option("UTC", help="Zone of timestamps without an offset, e.g. Europe/Warsaw"),
    timestamp_format: str = typer.Option(None, help="strptime format, e.g. '%d.%m.%Y %H:%M:%S'"),
    decimal: str = typer.Option(".", help="Decimal separator in CSV: '.' or ','"),
) -> None:
    """Import sensor readings from a file into the database (safe to repeat)."""
    settings = load_settings()
    try:
        load_machine(settings, machine_id)
        with DuckDBStorage(settings.db_path) as storage:
            report = ingest_file(
                storage,
                machine_id,
                file,
                layout=layout,
                timestamp_col=timestamp_col,
                sensor_col=sensor_col,
                value_col=value_col,
                tz=tz,
                timestamp_format=timestamp_format,
                decimal=decimal,
            )
        added = add_sensors(settings, machine_id, report.sensors)
    except (MachineError, IngestError) as exc:
        raise _fail(exc) from exc
    typer.echo(f"Imported {report.file} into {machine_id} ({report.layout} layout)")
    typer.echo(f"  rows in file:       {report.rows_in_file}")
    typer.echo(f"  readings found:     {report.readings_found}")
    typer.echo(f"  rejected (empty/invalid): {report.readings_rejected}")
    typer.echo(f"  new:                {report.readings_new}")
    typer.echo(f"  already stored:     {report.readings_duplicate}")
    typer.echo(f"  sensors:            {', '.join(report.sensors) or '-'}")
    if report.start and report.end:
        typer.echo(
            f"  range (UTC):        {report.start:%Y-%m-%d %H:%M} to {report.end:%Y-%m-%d %H:%M}"
        )
    if report.ignored_columns:
        typer.echo(f"  ignored columns:    {', '.join(report.ignored_columns)} (not numeric)")
    if added:
        typer.echo(f"  added to profile:   {', '.join(added)}")


@app.command("ingest-events")
def ingest_events_command(
    machine_id: str = typer.Argument(...),
    file: Path = typer.Argument(..., help="CSV or Parquet with failure timestamps and modes"),
    timestamp_col: str = typer.Option("timestamp"),
    mode_col: str = typer.Option("failure_mode"),
    description_col: str = typer.Option("description"),
    tz: str = typer.Option("UTC", help="Zone of timestamps without an offset"),
    timestamp_format: str = typer.Option(None, help="strptime format"),
    decimal: str = typer.Option("."),
) -> None:
    """Import labelled failures (needed later to train and evaluate prediction)."""
    settings = load_settings()
    try:
        load_machine(settings, machine_id)
        with DuckDBStorage(settings.db_path) as storage:
            report = ingest_events(
                storage,
                machine_id,
                file,
                timestamp_col=timestamp_col,
                mode_col=mode_col,
                description_col=description_col,
                tz=tz,
                timestamp_format=timestamp_format,
                decimal=decimal,
            )
    except (MachineError, IngestError) as exc:
        raise _fail(exc) from exc
    typer.echo(
        f"Imported {report.file}: {report.events_found} events found, "
        f"{report.events_new} new, {report.events_duplicate} already stored"
    )


@app.command("xlsx-template")
def xlsx_template(
    file: Path = typer.Argument(Path("diagsys_szablon.xlsx"), help="Where to write the template"),
) -> None:
    """Write an example Excel workbook (data, machine, sensors, failures, history, context)."""
    if file.exists():
        raise _fail(FileExistsError(f"{file} already exists; choose another name"))
    write_template(file)
    typer.echo(f"Wrote template: {file}")
    typer.echo(f"Fill it in, then: diagsys import-xlsx <machine-id> {file} --tz Europe/Warsaw")


@app.command("import-xlsx")
def import_xlsx(
    machine_id: str = typer.Argument(..., help="Machine id; created from the workbook if new"),
    file: Path = typer.Argument(..., help="Excel workbook (.xlsx), see: diagsys xlsx-template"),
    tz: str = typer.Option("UTC", help="Zone of timestamps without an offset, e.g. Europe/Warsaw"),
) -> None:
    """Import data, machine description, failure history and context from one Excel file."""
    settings = load_settings()
    try:
        with DuckDBStorage(settings.db_path) as storage:
            report = import_workbook(settings, storage, machine_id, file, tz=tz)
    except (MachineError, IngestError, XlsxError) as exc:
        raise _fail(exc) from exc
    r = report.readings
    typer.echo(f"Imported {report.file} into {report.machine_id}")
    if report.machine_created:
        typer.echo("  machine created from the workbook")
    typer.echo(
        f"  readings: {r.readings_found} found, {r.readings_new} new, "
        f"{r.readings_duplicate} already stored, {r.readings_rejected} rejected"
    )
    typer.echo(f"  sensors:  {', '.join(r.sensors) or '-'}")
    if r.start and r.end:
        typer.echo(f"  range (UTC): {r.start:%Y-%m-%d %H:%M} to {r.end:%Y-%m-%d %H:%M}")
    if report.events is not None:
        typer.echo(
            f"  failures: {report.events.events_found} found, {report.events.events_new} new"
        )
    if report.sensors_described or report.failure_modes:
        typer.echo(
            f"  profile:  {report.sensors_described} sensors and "
            f"{report.failure_modes} failure modes described"
        )
    if report.context_file:
        typer.echo(f"  context saved to {report.context_file}")
    for warning in report.warnings:
        typer.echo(f"  note: {warning}")


@data_app.command("summary")
def data_summary(machine_id: str = typer.Argument(...)) -> None:
    """Show what is stored for a machine: readings per sensor, time range, value range."""
    settings = load_settings()
    try:
        load_machine(settings, machine_id)
    except MachineError as exc:
        raise _fail(exc) from exc
    with DuckDBStorage(settings.db_path) as storage:
        rows = storage.summary(machine_id)
        events = storage.read_events(machine_id)
    if not rows:
        typer.echo(
            f"No readings for {machine_id}. Import some with: diagsys ingest {machine_id} <file>"
        )
    else:
        typer.echo(
            f"{'sensor':<22}{'readings':>10}  {'first (UTC)':<17}{'last (UTC)':<17}"
            f"{'min':>10}{'max':>10}{'mean':>10}"
        )
        for r in rows:
            typer.echo(
                f"{r.sensor_id:<22}{r.readings:>10}  {r.first:%Y-%m-%d %H:%M}   "
                f"{r.last:%Y-%m-%d %H:%M}  {r.minimum:>10.2f}{r.maximum:>10.2f}{r.mean:>10.2f}"
            )
    typer.echo(f"Failure events: {len(events)}")
    for event in events:
        typer.echo(f"  {event.timestamp:%Y-%m-%d %H:%M}  {event.failure_mode}  {event.description}")


# --- quality, cleaning, features (K3) ---------------------------------------------------------


def _prepare_command(
    machine_id: str,
    *,
    step_s: float | None,
    max_gap_s: float | None,
    flatline_points: int,
    spike_window: int,
    spike_threshold: float,
    max_interp: int,
    features: bool,
    window_s: float = 3600.0,
    stride_s: float = 600.0,
    min_valid_frac: float = 0.8,
    write_files: bool = True,
) -> None:
    settings = load_settings()
    try:
        profile = load_machine(settings, machine_id)
        with DuckDBStorage(settings.db_path) as storage:
            prepared = prepare(
                storage,
                profile,
                step_s=step_s,
                max_gap_s=max_gap_s,
                flatline_min_points=flatline_points,
                spike_window=spike_window,
                spike_threshold=spike_threshold,
                max_interp_points=max_interp,
                with_features=features,
                window_s=window_s,
                stride_s=stride_s,
                min_valid_frac=min_valid_frac,
            )
            report = prepared.quality.report
            ref = storage.write_result("quality_report", report)
    except (MachineError, FrameError, FeatureError, ValueError) as exc:
        raise _fail(exc) from exc
    _print_quality(report)
    typer.echo(f"Stored result: {ref}")
    if not write_files:
        return
    paths = write_outputs(settings, prepared, write_features=features)
    imputed = int(((prepared.clean.flags & FLAG_IMPUTED) != 0).sum())
    left = int(np.isnan(prepared.clean.frame.values).sum())
    typer.echo(f"Cleaning: {imputed} readings filled by interpolation, {left} left empty (NaN)")
    if prepared.features is not None:
        feats = prepared.features
        typer.echo(
            f"Features: {feats.num_rows} windows x {feats.num_columns - 5} features "
            f"(window {window_s:g} s, stride {stride_s:g} s)"
        )
    for name, path in paths.items():
        typer.echo(f"Wrote {name}: {path}")


def _print_quality(report: QualityReport) -> None:
    typer.echo(
        f"Data quality for {report.machine_id}: {report.start:%Y-%m-%d %H:%M} to "
        f"{report.end:%Y-%m-%d %H:%M} UTC, grid {report.step_s:g} s, {report.rows} rows, "
        f"{report.segments} continuous segments"
    )
    typer.echo(
        f"{'sensor':<20}{'observed':>10}{'missing':>9}{'%':>7}{'longest':>9}"
        f"{'spikes':>8}{'stuck':>7}{'range':>7}"
    )
    for q in report.sensors:
        pct = 100.0 * q.missing_points / q.expected_points if q.expected_points else 0.0
        typer.echo(
            f"{q.sensor_id:<20}{q.observed_points:>10}{q.missing_points:>9}{pct:>7.2f}"
            f"{q.longest_missing_run:>9}{q.spike_count:>8}{len(q.stuck_runs):>7}"
            f"{q.out_of_range_count:>7}"
        )
    typer.echo(
        f"Totals: {report.missing_points} missing, {report.spike_count} spikes, "
        f"{len(report.stuck_sensors)} stuck sensors, {report.gap_count} time gaps"
    )
    for gap in report.gaps[:10]:
        typer.echo(
            f"  gap {gap.start:%Y-%m-%d %H:%M} to {gap.end:%Y-%m-%d %H:%M} UTC "
            f"({gap.duration_s / 3600:.1f} h)"
        )
    if report.warnings:
        typer.echo("Warnings:")
        for warning in report.warnings:
            typer.echo(f"  - {warning}")


@app.command()
def quality(
    machine_id: str = typer.Argument(...),
    step_s: float = typer.Option(None, help="Grid step in seconds (default: typical interval)"),
    max_gap_s: float = typer.Option(None, help="Pause that splits data (default: 5 x step)"),
    flatline_points: int = typer.Option(20, min=3, help="Identical readings that mean 'stuck'"),
    spike_window: int = typer.Option(21, min=5, help="Rolling window of the spike filter"),
    spike_threshold: float = typer.Option(6.0, min=1.0, help="Spike limit in robust sigmas"),
    max_interp: int = typer.Option(5, min=0, help="Longest hole that cleaning fills"),
) -> None:
    """Check data quality: missing readings, spikes, stuck sensors, time gaps. Changes nothing."""
    _prepare_command(
        machine_id, step_s=step_s, max_gap_s=max_gap_s, flatline_points=flatline_points,
        spike_window=spike_window, spike_threshold=spike_threshold, max_interp=max_interp,
        features=False, write_files=False,
    )  # fmt: skip


@app.command()
def clean(
    machine_id: str = typer.Argument(...),
    step_s: float = typer.Option(None, help="Grid step in seconds (default: typical interval)"),
    max_gap_s: float = typer.Option(None, help="Pause that splits data (default: 5 x step)"),
    flatline_points: int = typer.Option(20, min=3, help="Identical readings that mean 'stuck'"),
    spike_window: int = typer.Option(21, min=5, help="Rolling window of the spike filter"),
    spike_threshold: float = typer.Option(6.0, min=1.0, help="Spike limit in robust sigmas"),
    max_interp: int = typer.Option(5, min=0, help="Longest hole that cleaning fills"),
) -> None:
    """Quality check + cleaning. Writes processed/clean.parquet and quality_report.json."""
    _prepare_command(
        machine_id, step_s=step_s, max_gap_s=max_gap_s, flatline_points=flatline_points,
        spike_window=spike_window, spike_threshold=spike_threshold, max_interp=max_interp,
        features=False,
    )  # fmt: skip


@app.command()
def features(
    machine_id: str = typer.Argument(...),
    window_s: float = typer.Option(3600.0, min=1.0, help="Window length in seconds"),
    stride_s: float = typer.Option(600.0, min=1.0, help="Step between windows in seconds"),
    min_valid_frac: float = typer.Option(0.8, min=0.0, max=1.0, help="Valid share a window needs"),
    step_s: float = typer.Option(None, help="Grid step in seconds (default: typical interval)"),
    max_gap_s: float = typer.Option(None, help="Pause that splits data (default: 5 x step)"),
    flatline_points: int = typer.Option(20, min=3, help="Identical readings that mean 'stuck'"),
    spike_window: int = typer.Option(21, min=5, help="Rolling window of the spike filter"),
    spike_threshold: float = typer.Option(6.0, min=1.0, help="Spike limit in robust sigmas"),
    max_interp: int = typer.Option(5, min=0, help="Longest hole that cleaning fills"),
) -> None:
    """Quality check, cleaning and window features. Writes report, clean and features files."""
    _prepare_command(
        machine_id, step_s=step_s, max_gap_s=max_gap_s, flatline_points=flatline_points,
        spike_window=spike_window, spike_threshold=spike_threshold, max_interp=max_interp,
        features=True, window_s=window_s, stride_s=stride_s, min_valid_frac=min_valid_frac,
    )  # fmt: skip
