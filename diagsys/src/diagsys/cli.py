"""Command line interface: doctor, machine, simulate, ingest, ingest-events, data summary,
quality, clean, features, import-xlsx, xlsx-template, diagnose, experiment-diagnosis, predict,
experiment-prediction, llm check, docs index/search/ask, ui."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import typer
import yaml

from . import __version__
from .config import load_settings
from .contracts import ChatMessage, QualityReport
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
from .eval import RunError, run_diagnosis_experiment, run_prediction_experiment
from .gateway import SimConfig, write_simulation
from .gateway import simulate as run_simulation
from .llm import LLMError, OpenAICompatClient
from .machines import (
    MachineError,
    add_sensors,
    create_machine,
    list_machines,
    load_machine,
    machine_dir,
)
from .rag import (
    RagError,
    answer_question,
    index_machine,
    make_embedder,
    search_machine,
)
from .storage import DuckDBStorage
from .tools import (
    METHODS,
    RUL_METHODS,
    DiagnosisError,
    PredictionError,
    build_dataset,
    diagnose,
    fit_model,
    forecast_latest,
    to_anomaly_report,
)

app = typer.Typer(
    help="Universal local multi-agent system for machine diagnostics.",
    no_args_is_help=True,
    add_completion=False,
)
machine_app = typer.Typer(help="Machine profiles.", no_args_is_help=True)
data_app = typer.Typer(help="Inspect stored data.", no_args_is_help=True)
app.add_typer(machine_app, name="machine")
llm_app = typer.Typer(help="Model server (Bielik via Ollama or llama.cpp).", no_args_is_help=True)
docs_app = typer.Typer(help="Machine documentation: index, search, ask.", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(llm_app, name="llm")
app.add_typer(docs_app, name="docs")

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


# --- diagnosis (K4) ---------------------------------------------------------------------------


def _features_for(settings, machine_id: str, window_s: float, stride_s: float):
    """Window features recomputed from the stored readings (same snapshot as the failures)."""
    profile = load_machine(settings, machine_id)
    storage = DuckDBStorage(settings.db_path)
    try:
        prepared = prepare(
            storage, profile, with_features=True, window_s=window_s, stride_s=stride_s
        )
    except Exception:
        storage.close()
        raise
    return profile, storage, prepared.features


@app.command("diagnose")
def diagnose_command(
    machine_id: str = typer.Argument(...),
    method: str = typer.Option("iforest", help=f"One of: {', '.join(METHODS)}"),
    exclude_sensor: list[str] = typer.Option(
        None, "--exclude-sensor", help="Sensor to leave out, e.g. a load sensor (repeatable)"
    ),
    reference_windows: int = typer.Option(
        144, help="Healthy start of each segment, in windows (0 = a fraction of the segment)"
    ),
    reference_frac: float = typer.Option(0.25, help="Fraction used when --reference-windows is 0"),
    window_s: float = typer.Option(3600.0, help="Feature window length in seconds"),
    stride_s: float = typer.Option(600.0, help="Step between windows in seconds"),
    seed: int = typer.Option(42),
) -> None:
    """Find anomalous periods and the sensors behind them (needs imported data)."""
    settings = load_settings()
    try:
        profile, storage, features = _features_for(settings, machine_id, window_s, stride_s)
        with storage:
            result = diagnose(
                features,
                method=method,
                exclude_sensors=tuple(exclude_sensor or ()),
                reference_frac=reference_frac,
                reference_windows=reference_windows or None,
                seed=seed,
            )
    except (MachineError, FrameError, FeatureError, DiagnosisError) as exc:
        raise _fail(exc) from exc
    report = to_anomaly_report(result, machine_id)
    out = machine_dir(settings, machine_id) / "processed"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"diagnosis_{method}.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(
        f"Diagnosis of {machine_id} ({method}): {len(result.episodes)} alarm episodes in "
        f"{result.monitored_hours:.0f} h monitored (threshold {result.threshold:.2f})"
    )
    if result.skipped_segments:
        typer.echo(f"  segments too short to diagnose: {result.skipped_segments}")
    for ep in result.episodes:
        drivers = "; ".join(f"{sensor} {stat} {z:+.1f}" for sensor, stat, z, _ in ep.drivers[:3])
        typer.echo(
            f"  detected {ep.detected_at:%Y-%m-%d %H:%M}  to {ep.end:%Y-%m-%d %H:%M}  "
            f"({ep.n_windows} windows)  {drivers}"
        )
    typer.echo(f"Saved: {path}")


@app.command("experiment-diagnosis")
def experiment_diagnosis(
    machine_id: str = typer.Argument(...),
    horizon_h: float = typer.Option(
        72.0, help="An alarm counts if it comes this long before a failure"
    ),
    exclude_sensor: list[str] = typer.Option(None, "--exclude-sensor"),
    seed: int = typer.Option(42),
    name: str = typer.Option("diagnosis", help="Run name (letters, digits, - and _)"),
) -> None:
    """Compare the diagnosis methods against the known failures; stores a reproducible run."""
    settings = load_settings()
    try:
        profile, storage, features = _features_for(settings, machine_id, 3600.0, 600.0)
        with storage:
            run, scores = run_diagnosis_experiment(
                settings,
                storage,
                profile,
                features,
                horizon_h=horizon_h,
                seed=seed,
                diagnose_options={"exclude_sensors": tuple(exclude_sensor or ())},
                name=name,
            )
    except (MachineError, FrameError, FeatureError, DiagnosisError, RunError) as exc:
        raise _fail(exc) from exc
    typer.echo(f"Run {run.manifest.run_id} (seed {seed}, alarm horizon {horizon_h:g} h)")
    typer.echo(
        f"{'method':<10}{'detected':>10}{'median lead':>13}{'false alarms':>14}{'per 100 h':>11}"
    )
    for sc in scores:
        lead = f"{sc.median_lead_h:.1f} h" if sc.median_lead_h is not None else "-"
        rate = f"{sc.false_alarms_per_100h:.2f}" if sc.false_alarms_per_100h is not None else "-"
        typer.echo(
            f"{sc.method:<10}{f'{sc.detected}/{sc.failures}':>10}{lead:>13}"
            f"{sc.false_alarms:>14}{rate:>11}"
        )
    typer.echo(f"Saved in {run.path}")


# --- prediction (K5) --------------------------------------------------------------------------


def _parse_at(text: str) -> datetime:
    try:
        moment = datetime.fromisoformat(text)
    except ValueError as exc:
        raise typer.BadParameter(f"not a date and time: {text!r} (e.g. 2026-03-01T12:00)") from exc
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


@app.command("predict")
def predict_command(
    machine_id: str = typer.Argument(...),
    method: str = typer.Option("quantile_gbm", help=f"One of: {', '.join(RUL_METHODS)}"),
    horizon_h: float = typer.Option(72.0, help="Horizon of the failure probability, hours"),
    exclude_sensor: list[str] = typer.Option(None, "--exclude-sensor"),
    at: str = typer.Option(
        None,
        "--at",
        help="Forecast as of this moment (UTC unless an offset is given), using only data "
        "and failures up to then: an honest backtest",
    ),
    seed: int = typer.Option(42),
) -> None:
    """Forecast the remaining useful life from the recorded history and failures."""
    settings = load_settings()
    try:
        as_of = _parse_at(at) if at else None
        profile, storage, features = _features_for(settings, machine_id, 3600.0, 600.0)
        with storage:
            failures = [e.timestamp for e in storage.read_events(machine_id)]
        if as_of is not None:
            cutoff = pa.scalar(as_of, pa.timestamp("us", tz="UTC"))
            features = features.filter(pc.less_equal(features.column("window_end"), cutoff))
            failures = [f for f in failures if f <= as_of]
        ds = build_dataset(features, failures, exclude_sensors=tuple(exclude_sensor or ()))
        model = fit_model(method, ds, seed=seed)
        result = forecast_latest(model, ds, machine_id, horizon_h=horizon_h)
    except (MachineError, FrameError, FeatureError, PredictionError) as exc:
        raise _fail(exc) from exc
    f = result.forecast
    typer.echo(f"Forecast for {machine_id} as of {result.as_of:%Y-%m-%d %H:%M} UTC ({method})")
    typer.echo(f"  time since last repair: {result.age_h:.0f} h")
    typer.echo(
        f"  remaining useful life: {f.rul_hours:.0f} h  "
        f"(80% interval {f.rul_low_hours:.0f} to {f.rul_high_hours:.0f} h)"
    )
    typer.echo(
        f"  probability of failure within {f.horizon_hours:g} h: {f.failure_probability:.0%} "
        "(approximate)"
    )
    typer.echo(
        f"  trained on {model.n_train_cycles} failure cycles ({model.n_train_windows} windows)"
    )
    for warning in result.warnings:
        typer.echo(f"  WARNING: {warning}")
    out = machine_dir(settings, machine_id) / "processed"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"forecast_{method}.json"
    path.write_text(f.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(f"Saved: {path}")


@app.command("experiment-prediction")
def experiment_prediction(
    machine_id: str = typer.Argument(...),
    horizon_h: float = typer.Option(72.0),
    exclude_sensor: list[str] = typer.Option(None, "--exclude-sensor"),
    seed: int = typer.Option(42),
    name: str = typer.Option("prediction", help="Run name (letters, digits, - and _)"),
) -> None:
    """Leave-one-cycle-out comparison of the prediction methods; stores a reproducible run."""
    settings = load_settings()
    try:
        profile, storage, features = _features_for(settings, machine_id, 3600.0, 600.0)
        with storage:
            run, scores = run_prediction_experiment(
                settings,
                storage,
                profile,
                features,
                horizon_h=horizon_h,
                seed=seed,
                dataset_options={"exclude_sensors": tuple(exclude_sensor or ())},
                name=name,
            )
    except (MachineError, FrameError, FeatureError, PredictionError, RunError) as exc:
        raise _fail(exc) from exc
    typer.echo(f"Run {run.manifest.run_id} (seed {seed}, horizon {horizon_h:g} h)")
    typer.echo(
        f"{'method':<14}{'MAE':>8}{'MAE<=72h':>10}{'cover80':>9}{'width':>8}{'Brier skill':>13}"
    )
    for sc in scores:
        late = f"{sc.mae_last72_h:.1f}h" if sc.mae_last72_h is not None else "-"
        skill = f"{sc.brier_skill:.2f}" if sc.brier_skill is not None else "-"
        typer.echo(
            f"{sc.method:<14}{sc.mae_h:>7.1f}h{late:>10}{sc.coverage80:>9.2f}"
            f"{sc.mean_width_h:>7.0f}h{skill:>13}"
        )
    typer.echo(f"Saved in {run.path}")


# --- model server and documentation (K6) --------------------------------------------------------


@llm_app.command("check")
def llm_check() -> None:
    """Check that the model server answers, and that the configured models are available."""
    settings = load_settings()
    try:
        client = OpenAICompatClient.from_settings(settings)
        names = client.models()
        typer.echo(
            f"Server {settings.llm_base_url}: {len(names)} models: {', '.join(names) or '-'}"
        )
        for role, wanted in (("chat", settings.llm_model), ("embedding", settings.embedding_model)):
            if role == "embedding" and settings.embedding_backend != "server":
                continue
            if not any(n == wanted or n.startswith(wanted + ":") for n in names):
                typer.echo(f"  WARNING: {role} model {wanted!r} is not in the list above")
        reply = client.chat(
            [ChatMessage(role="user", content="Odpowiedz jednym słowem: OK")], max_tokens=16
        )
        typer.echo(f"  chat model {settings.llm_model!r} answered: {reply[:80]!r}")
        if settings.embedding_backend == "server":
            dim = len(client.embed(["test"])[0])
            typer.echo(f"  embedding model {settings.embedding_model!r}: vectors of size {dim}")
        else:
            typer.echo(
                "  embeddings: local 'hash' backend (DIAGSYS_EMBEDDING_BACKEND=server to change)"
            )
    except LLMError as exc:
        raise _fail(exc) from exc


@app.command("ui")
def ui_command(
    port: int = typer.Option(8501, help="Port of the local web page"),
) -> None:
    """Open the supervisor window in the browser (listens on this computer only)."""
    import subprocess
    import sys

    try:
        import streamlit  # noqa: F401
    except ImportError as exc:
        raise _fail(
            RuntimeError(
                "the window needs Streamlit: uv sync --extra app (or pip install streamlit)"
            )
        ) from exc
    page = Path(__file__).parent / "ui" / "app.py"
    command = [
        sys.executable, "-m", "streamlit", "run", str(page),
        "--server.address", "127.0.0.1", "--server.port", str(port),
        "--browser.gatherUsageStats", "false",
    ]  # fmt: skip
    raise typer.Exit(subprocess.call(command))  # noqa: S603


@docs_app.command("index")
def docs_index(
    machine_id: str = typer.Argument(...),
    rebuild: bool = typer.Option(False, help="Forget the index and build it again"),
) -> None:
    """Index the files in the machine's docs/ folder (.txt, .md, .pdf, .docx); incremental."""
    settings = load_settings()
    try:
        embedder = make_embedder(settings)
        report = index_machine(settings, machine_id, embedder, rebuild=rebuild)
    except (MachineError, RagError, LLMError) as exc:
        raise _fail(exc) from exc
    typer.echo(f"Indexed {machine_id} into {report.collection} (embedder {embedder.id})")
    typer.echo(
        f"  files: {len(report.added)} new, {len(report.updated)} changed, "
        f"{len(report.removed)} removed, {len(report.unchanged)} unchanged"
    )
    typer.echo(f"  passages: {report.passages_added} added, {report.passages_total} in total")
    for name, reason in report.skipped:
        typer.echo(f"  skipped {name}: {reason}")
    if report.passages_total == 0:
        typer.echo("  nothing to search yet: put manuals into the machine's docs/ folder")


@docs_app.command("search")
def docs_search(
    machine_id: str = typer.Argument(...),
    query: str = typer.Argument(..., help="What to look for"),
    k: int = typer.Option(5, "-k", "--top", help="Number of passages"),
) -> None:
    """Find the passages of the machine's documentation that best match the query."""
    settings = load_settings()
    try:
        passages = search_machine(settings, machine_id, make_embedder(settings), query, k)
    except (MachineError, RagError, LLMError) as exc:
        raise _fail(exc) from exc
    if not passages:
        typer.echo("No passages found. Run: diagsys docs index " + machine_id)
        return
    for i, p in enumerate(passages, start=1):
        where = p.doc_id + (f", p. {p.page}" if p.page else "")
        typer.echo(f"[{i}] {p.score:.2f}  {where}")
        typer.echo("    " + " ".join(p.text.split())[:240])


@docs_app.command("ask")
def docs_ask(
    machine_id: str = typer.Argument(...),
    question: str = typer.Argument(...),
    k: int = typer.Option(5, "-k", "--top", help="Number of passages given to the model"),
) -> None:
    """Answer a question from the machine's documentation, with cited sources (needs the model)."""
    settings = load_settings()
    try:
        passages = search_machine(settings, machine_id, make_embedder(settings), question, k)
        answer = answer_question(OpenAICompatClient.from_settings(settings), question, passages)
    except (MachineError, RagError, LLMError) as exc:
        raise _fail(exc) from exc
    typer.echo(answer.text)
    for n, source in zip(answer.citations, answer.sources, strict=True):
        typer.echo(f"  [{n}] {source.ref}")
    for warning in answer.warnings:
        typer.echo(f"WARNING: {warning}")
