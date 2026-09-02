from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Annotated, Any, Literal, TextIO

from pydantic import TypeAdapter, ValidationError
import torch
import typer
import yaml

from .artifacts import ArtifactRepository
from .contracts import (
    AttentionHeadInterventionSpec,
    AttentionHeadRankSpec,
    AttentionTraceSpec,
    ErrorDetail,
    ErrorEnvelope,
    ExecutionReceipt,
    DirectionInjectionSpec,
    ExperimentSpec,
    FFNCouplingSpec,
    InterventionSpec,
    JobEvent,
    ModelRequest,
    QualificationSpec,
    RankSpec,
    ResearchWorkflowSpec,
    RunManifest,
    TrajectorySpec,
)
from .errors import EndpointError, ProbeError, SpecError
from .legacy import DEFAULT_MODEL, PROJECT_DIRECTORY, main as legacy_main
from .models import ModelManager
from .reporting import build_overview, query_envelope
from .service import ResearchService
from .specs import (
    canonical_json,
    example_rank_spec,
    example_replication_spec,
    load_spec,
    request_hash,
    science_hash,
)


DEFAULT_WORKSPACE = PROJECT_DIRECTORY / ".probe"
DEFAULT_CACHE = PROJECT_DIRECTORY / ".hf-cache"
OutputMode = Literal["human", "json", "jsonl"]
JSONABLE_ADAPTER = TypeAdapter(Any)


@dataclass(frozen=True)
class CLIContext:
    workspace: Path
    cache_dir: Path
    endpoint: str | None
    token_file: Path | None


app = typer.Typer(
    name="probe",
    help="Local-first perturbation-probing research platform.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
schema_app = typer.Typer(help="Discover versioned machine schemas.")
examples_app = typer.Typer(help="Inspect canonical experiment examples.")
model_app = typer.Typer(help="Inspect and acquire model snapshots.")
jobs_app = typer.Typer(help="Inspect and control research jobs.")
runs_app = typer.Typer(help="Query and verify immutable run artifacts.")
server_app = typer.Typer(help="Manage the local model daemon.")
replay_app = typer.Typer(help="Record and reproduce portable experiment bundles.")
attention_app = typer.Typer(help="Rank, intervene on, and trace attention routes.")
app.add_typer(schema_app, name="schema")
app.add_typer(examples_app, name="examples")
app.add_typer(model_app, name="model")
app.add_typer(jobs_app, name="jobs")
app.add_typer(runs_app, name="runs")
app.add_typer(server_app, name="server")
app.add_typer(replay_app, name="replay")
app.add_typer(attention_app, name="attention")


@app.callback()
def root(
    ctx: typer.Context,
    workspace: Annotated[Path, typer.Option(help="Files-only job and run store.")] = DEFAULT_WORKSPACE,
    cache_dir: Annotated[Path, typer.Option(help="Project-local Hugging Face cache.")] = DEFAULT_CACHE,
    endpoint: Annotated[str | None, typer.Option(help="Explicit daemon URL.")] = None,
    token_file: Annotated[Path | None, typer.Option(help="Daemon bearer-token file.")] = None,
) -> None:
    ctx.obj = CLIContext(
        workspace=workspace,
        cache_dir=cache_dir,
        endpoint=endpoint.rstrip("/") if endpoint else None,
        token_file=token_file,
    )


def _context(ctx: typer.Context) -> CLIContext:
    return ctx.ensure_object(CLIContext)


def _dump(value: Any, *, mode: OutputMode = "json") -> None:
    # Handles containers of Pydantic records as well as a model at the root.
    # This matters for agent-facing commands such as `runs files`, whose
    # repository result is a tuple[ArtifactRef, ...].
    value = JSONABLE_ADAPTER.dump_python(value, mode="json")
    if mode == "human":
        typer.echo(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        typer.echo(canonical_json(value))


def _error_detail(exc: Exception) -> tuple[ErrorDetail, int]:
    if isinstance(exc, ProbeError):
        return exc.as_detail(), exc.exit_code
    if isinstance(exc, (ValidationError, ValueError, json.JSONDecodeError, yaml.YAMLError)):
        return (
            ErrorDetail(
                code="invalid_spec",
                message=str(exc),
                hint="Run 'probe schema show rank' and validate the input document.",
            ),
            2,
        )
    return (
        ErrorDetail(
            code="runtime_error",
            message=str(exc),
            details={"exception_type": type(exc).__name__},
        ),
        5,
    )


def _fail(exc: Exception, *, machine: bool = False) -> None:
    detail, exit_code = _error_detail(exc)
    envelope = ErrorEnvelope(error=detail)
    target = sys.stdout if machine else sys.stderr
    target.write(canonical_json(envelope) + "\n" if machine else f"Error: {detail.message}\n")
    raise typer.Exit(exit_code)


def _service(context: CLIContext) -> ResearchService:
    return ResearchService(workspace=context.workspace, cache_dir=context.cache_dir)


def _event_listener(mode: Literal["human", "jsonl", "none"]):
    output = sys.stdout

    def listener(event: JobEvent) -> None:
        if mode == "none":
            return
        if mode == "jsonl":
            output.write(canonical_json(event) + "\n")
            output.flush()
            return
        if event.event in {
            "model.loading",
            "model.reused",
            "model.ready",
            "pair.started",
            "pair.completed",
            "qualify.started",
            "qualify.completed",
            "intervention.started",
            "intervention.completed",
            "direction.started",
            "direction.completed",
            "attention_rank.started",
            "attention_rank.completed",
            "attention_intervention.started",
            "attention_intervention.completed",
            "attention_trace.started",
            "attention_trace.completed",
            "warning",
            "artifact.committed",
            "job.completed",
        }:
            typer.echo(f"[{event.event}] {json.dumps(event.payload, ensure_ascii=False)}")

    return listener


@app.command()
def doctor(
    ctx: typer.Context,
    output: Annotated[OutputMode, typer.Option()] = "human",
) -> None:
    context = _context(ctx)
    packages = {}
    for name in ("torch", "transformers", "pydantic", "fastapi", "typer", "textual"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    value = {
        "schema_version": "probe.doctor/v1",
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "mps_built": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built()),
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "cuda_available": torch.cuda.is_available(),
        "workspace": str(context.workspace.resolve()),
        "cache_dir": str(context.cache_dir.resolve()),
        "endpoint": context.endpoint,
        "packages": packages,
    }
    _dump(value, mode=output)


@schema_app.command("list")
def schema_list() -> None:
    _dump(
        {
            "schemas": [
                "rank",
                "trajectory",
                "ffn-coupling",
                "qualification",
                "intervention",
                "direction",
                "attention-rank",
                "attention-intervention",
                "attention-trace",
                "qualification-result",
                "intervention-result",
                "direction-result",
                "attention-rank-result",
                "attention-intervention-result",
                "attention-trace-result",
                "experiment-set",
                "perturbation-template",
                "perturbation-compilation",
                "comparison",
                "stability",
                "sensitivity",
                "workflow",
                "workflow-outcome",
                "research-case",
                "research-case-create",
                "research-case-update",
                "research-case-plan",
                "research-report",
                "report-receipt",
                "plan",
                "preflight",
                "execution-receipt",
                "capabilities",
                "result",
                "trajectory-result",
                "ffn-coupling-result",
                "overview",
                "query",
                "verification",
                "replay-driver",
                "replay-baseline",
                "replay-report",
                "replay-outcome",
                "replay-identity",
                "replay-record",
                "job",
                "event",
                "run",
                "error",
            ]
        }
    )


@schema_app.command("show")
def schema_show(name: str) -> None:
    from .contracts import (
        AttentionHeadInterventionRunSummary,
        AttentionHeadRankRunSummary,
        AttentionTraceRunSummary,
        CapabilityReport,
        ComparisonReport,
        DirectionInjectionRunSummary,
        DirectionInjectionSpec,
        ErrorEnvelope,
        ExperimentPlan,
        ExecutionReceipt,
        ExperimentSet,
        FFNCouplingRunSummary,
        FFNCouplingSpec,
        JobEvent,
        JobStatus,
        InterventionRunSummary,
        InterventionSpec,
        PreflightReport,
        PerturbationCompilation,
        PerturbationTemplate,
        QualificationRunSummary,
        QualificationSpec,
        QueryEnvelope,
        RankRunSummary,
        ReportReceipt,
        ResearchReport,
        ResearchCase,
        ResearchCaseCreate,
        ResearchCasePlan,
        ResearchCaseUpdate,
        ResearchWorkflowOutcome,
        ResearchWorkflowSpec,
        ReplayBaseline,
        ReplayDriver,
        ReplayIdentity,
        ReplayOutcome,
        ReplayRecordReceipt,
        ReplayReport,
        RunOverview,
        RunManifest,
        TrajectoryRunSummary,
        TrajectorySpec,
        SensitivityReport,
        StabilityReport,
        VerificationReport,
    )

    schemas = {
        "rank": RankSpec,
        "trajectory": TrajectorySpec,
        "ffn-coupling": FFNCouplingSpec,
        "qualification": QualificationSpec,
        "intervention": InterventionSpec,
        "direction": DirectionInjectionSpec,
        "attention-rank": AttentionHeadRankSpec,
        "attention-intervention": AttentionHeadInterventionSpec,
        "attention-trace": AttentionTraceSpec,
        "qualification-result": QualificationRunSummary,
        "intervention-result": InterventionRunSummary,
        "direction-result": DirectionInjectionRunSummary,
        "attention-rank-result": AttentionHeadRankRunSummary,
        "attention-intervention-result": AttentionHeadInterventionRunSummary,
        "attention-trace-result": AttentionTraceRunSummary,
        "trajectory-result": TrajectoryRunSummary,
        "ffn-coupling-result": FFNCouplingRunSummary,
        "experiment-set": ExperimentSet,
        "perturbation-template": PerturbationTemplate,
        "perturbation-compilation": PerturbationCompilation,
        "comparison": ComparisonReport,
        "stability": StabilityReport,
        "sensitivity": SensitivityReport,
        "workflow": ResearchWorkflowSpec,
        "workflow-outcome": ResearchWorkflowOutcome,
        "research-case": ResearchCase,
        "research-case-create": ResearchCaseCreate,
        "research-case-update": ResearchCaseUpdate,
        "research-case-plan": ResearchCasePlan,
        "research-report": ResearchReport,
        "report-receipt": ReportReceipt,
        "plan": ExperimentPlan,
        "preflight": PreflightReport,
        "execution-receipt": ExecutionReceipt,
        "capabilities": CapabilityReport,
        "result": RankRunSummary,
        "overview": RunOverview,
        "query": QueryEnvelope,
        "verification": VerificationReport,
        "replay-driver": ReplayDriver,
        "replay-baseline": ReplayBaseline,
        "replay-report": ReplayReport,
        "replay-outcome": ReplayOutcome,
        "replay-identity": ReplayIdentity,
        "replay-record": ReplayRecordReceipt,
        "job": JobStatus,
        "event": JobEvent,
        "run": RunManifest,
        "error": ErrorEnvelope,
    }
    try:
        model = schemas[name]
    except KeyError:
        _fail(SpecError(f"unknown schema {name!r}; choices: {sorted(schemas)}"), machine=True)
        return
    _dump(model.model_json_schema())


@examples_app.command("list")
def examples_list() -> None:
    _dump({"examples": ["agreement-capital", "agreement-replication"]})


@examples_app.command("show")
def examples_show(
    name: Annotated[str, typer.Argument()] = "agreement-capital",
    format: Annotated[Literal["json", "yaml"], typer.Option()] = "json",
) -> None:
    examples = {
        "agreement-capital": example_rank_spec,
        "agreement-replication": example_replication_spec,
    }
    if name not in examples:
        _fail(SpecError(f"unknown example {name!r}"), machine=True)
        return
    value = examples[name]().model_dump(mode="json")
    if format == "yaml":
        typer.echo(yaml.safe_dump(value, sort_keys=False, allow_unicode=True))
    else:
        _dump(value)


@model_app.command("inspect")
def model_inspect(
    ctx: typer.Context,
    model_id: Annotated[str, typer.Argument()] = DEFAULT_MODEL,
    revision: Annotated[str | None, typer.Option()] = None,
) -> None:
    manager = ModelManager(_context(ctx).cache_dir)
    _dump(manager.inspect_cached(ModelRequest(id=model_id, revision=revision)))


@model_app.command("fetch")
def model_fetch(
    ctx: typer.Context,
    model_id: Annotated[str, typer.Argument()] = DEFAULT_MODEL,
    revision: Annotated[str | None, typer.Option()] = None,
    max_download_bytes: Annotated[int, typer.Option(min=1)] = 2_000_000_000,
) -> None:
    try:
        path = ModelManager(_context(ctx).cache_dir).fetch(
            ModelRequest(id=model_id, revision=revision),
            max_download_bytes=max_download_bytes,
        )
        _dump({"model_id": model_id, "path": str(path), "cached": True})
    except Exception as exc:
        _fail(exc, machine=True)


@app.command()
def validate(
    spec: Annotated[str, typer.Option(help="JSON/YAML file, or '-' for stdin.")],
) -> None:
    try:
        parsed = load_spec(spec)
        _dump(
            {
                "schema_version": "probe.validation/v1",
                "valid": True,
                "science_hash": science_hash(parsed),
                "request_hash": request_hash(parsed),
            }
        )
    except Exception as exc:
        _fail(exc, machine=True)


@app.command("perturb")
def perturb_command(
    template: Annotated[str, typer.Option(help="Perturbation template JSON/YAML file.")],
) -> None:
    from .contracts import PerturbationTemplate
    from .perturbations import compile_perturbations
    from .specs import load_document

    try:
        parsed = PerturbationTemplate.model_validate(load_document(template))
        _dump(compile_perturbations(parsed))
    except Exception as exc:
        _fail(exc, machine=True)


@app.command()
def plan(
    ctx: typer.Context,
    spec: Annotated[str, typer.Option(help="JSON/YAML file, or '-' for stdin.")],
) -> None:
    try:
        parsed = load_spec(spec)
        context = _context(ctx)
        if context.endpoint:
            from .client import ProbeClient

            _dump(ProbeClient.from_context(context).plan(parsed))
        else:
            _dump(_service(context).plan(parsed))
    except Exception as exc:
        _fail(exc, machine=True)


@app.command()
def preflight(
    ctx: typer.Context,
    spec: Annotated[str, typer.Option(help="JSON/YAML file, or '-' for stdin.")],
) -> None:
    try:
        parsed = load_spec(spec)
        context = _context(ctx)
        if context.endpoint:
            from .client import ProbeClient

            value = ProbeClient.from_context(context).preflight(parsed)
        else:
            value = _service(context).preflight(parsed)
        _dump(value)
    except Exception as exc:
        _fail(exc, machine=True)


@app.command()
def capabilities(
    ctx: typer.Context,
    spec: Annotated[str, typer.Option(help="JSON/YAML file, or '-' for stdin.")],
) -> None:
    try:
        parsed = load_spec(spec)
        context = _context(ctx)
        if context.endpoint:
            from .client import ProbeClient

            _dump(ProbeClient.from_context(context).capabilities(parsed))
        else:
            _dump(_service(context).capabilities(parsed))
    except Exception as exc:
        _fail(exc, machine=True)


def _execute_parsed(
    *,
    context: CLIContext,
    parsed: ExperimentSpec,
    events: Literal["human", "jsonl", "none"],
    request_id: str | None,
    diagnostic_stream: TextIO | None,
) -> str:
    if context.endpoint:
        from .client import ProbeClient

        status = ProbeClient.from_context(context).run(
            parsed,
            listener=_event_listener(events),
            request_id=request_id,
        )
        run_id = status.get("run_id")
        if not run_id:
            raise EndpointError("daemon job completed without a run ID")
        return str(run_id)
    outcome = _service(context).execute(
        parsed,
        listener=_event_listener(events),
        diagnostic_stream=diagnostic_stream,
    )
    return outcome.manifest.run_id


def _compact_result(context: CLIContext, run_id: str) -> Any:
    if context.endpoint:
        from .client import ProbeClient

        manifest = ProbeClient.from_context(context).run_manifest(run_id)
        run_kind = manifest.get("run_kind", "rank")
    else:
        run_kind = ArtifactRepository(context.workspace).load_manifest(run_id).run_kind
    if run_kind == "rank":
        return _overview(context, run_id)
    return _summary(context, run_id)


def _execution_receipt(context: CLIContext, run_id: str) -> ExecutionReceipt:
    if context.endpoint:
        from .client import ProbeClient

        manifest = ProbeClient.from_context(context).run_manifest(run_id)
        run_kind = manifest["run_kind"]
        evidence_stage = manifest["evidence_stage"]
    else:
        manifest = ArtifactRepository(context.workspace).load_manifest(run_id)
        run_kind = manifest.run_kind
        evidence_stage = manifest.evidence_stage
    result = JSONABLE_ADAPTER.dump_python(
        _compact_result(context, run_id), mode="json"
    )
    return ExecutionReceipt(
        run_id=run_id,
        run_kind=run_kind,
        evidence_stage=evidence_stage,
        logical_forward_passes=int(result["logical_forward_passes"]),
        result=result,
    )


@app.command()
def run(
    ctx: typer.Context,
    spec: Annotated[str, typer.Option(help="JSON/YAML file, or '-' for stdin.")],
    events: Annotated[Literal["human", "jsonl", "none"], typer.Option()] = "human",
    result: Annotated[Literal["none", "compact-json"], typer.Option()] = "none",
    request_id: Annotated[
        str | None,
        typer.Option(help="Idempotency key; requires an explicit daemon endpoint."),
    ] = None,
) -> None:
    try:
        if result == "compact-json" and events != "none":
            raise SpecError("--result compact-json requires --events none")
        parsed = load_spec(spec)
        context = _context(ctx)
        if request_id is not None and context.endpoint is None:
            raise SpecError("--request-id requires an explicit daemon endpoint")
        if request_id is not None:
            request_id = request_id.strip()
            if not request_id or len(request_id) > 256:
                raise SpecError(
                    "--request-id must contain 1 to 256 non-whitespace characters"
                )
        run_id = _execute_parsed(
            context=context,
            parsed=parsed,
            events=events,
            request_id=request_id,
            diagnostic_stream=(
                sys.stderr
                if events == "jsonl" or result == "compact-json"
                else None
            ),
        )
        if result == "compact-json":
            _dump(_compact_result(context, run_id))
    except Exception as exc:
        _fail(exc, machine=events == "jsonl" or result == "compact-json")


def _typed_experiment_command(
    *,
    ctx: typer.Context,
    spec_path: str,
    expected: (
        type[RankSpec]
        | type[QualificationSpec]
        | type[InterventionSpec]
        | type[DirectionInjectionSpec]
        | type[AttentionHeadRankSpec]
        | type[AttentionHeadInterventionSpec]
        | type[AttentionTraceSpec]
    ),
    events: Literal["human", "jsonl", "none"],
) -> None:
    try:
        parsed = load_spec(spec_path)
        if not isinstance(parsed, expected):
            raise SpecError(
                f"expected kind {expected.model_fields['kind'].default!r}, got {parsed.kind!r}"
            )
        context = _context(ctx)
        run_id = _execute_parsed(
            context=context,
            parsed=parsed,
            events=events,
            request_id=None,
            diagnostic_stream=sys.stderr,
        )
        _dump(
            _execution_receipt(context, run_id),
            mode="jsonl" if events == "jsonl" else "json",
        )
    except Exception as exc:
        _fail(exc, machine=True)


@app.command("rank")
def rank_command(
    ctx: typer.Context,
    spec: Annotated[str, typer.Option(help="Rank JSON/YAML file, or '-' for stdin.")],
    events: Annotated[Literal["human", "jsonl", "none"], typer.Option()] = "none",
) -> None:
    _typed_experiment_command(
        ctx=ctx, spec_path=spec, expected=RankSpec, events=events
    )


@app.command("qualify")
def qualify_command(
    ctx: typer.Context,
    spec: Annotated[
        str, typer.Option(help="Qualification JSON/YAML file, or '-' for stdin.")
    ],
    events: Annotated[Literal["human", "jsonl", "none"], typer.Option()] = "none",
) -> None:
    _typed_experiment_command(
        ctx=ctx, spec_path=spec, expected=QualificationSpec, events=events
    )


@app.command("intervene")
def intervene_command(
    ctx: typer.Context,
    spec: Annotated[
        str, typer.Option(help="Intervention JSON/YAML file, or '-' for stdin.")
    ],
    events: Annotated[Literal["human", "jsonl", "none"], typer.Option()] = "none",
) -> None:
    _typed_experiment_command(
        ctx=ctx, spec_path=spec, expected=InterventionSpec, events=events
    )


@app.command("inject")
def inject_command(
    ctx: typer.Context,
    spec: Annotated[
        str, typer.Option(help="Direction-injection JSON/YAML file, or '-' for stdin.")
    ],
    events: Annotated[Literal["human", "jsonl", "none"], typer.Option()] = "none",
) -> None:
    _typed_experiment_command(
        ctx=ctx, spec_path=spec, expected=DirectionInjectionSpec, events=events
    )


@attention_app.command("rank")
def attention_rank_command(
    ctx: typer.Context,
    spec: Annotated[str, typer.Option(help="Attention-rank JSON/YAML spec.")],
    events: Annotated[Literal["human", "jsonl", "none"], typer.Option()] = "none",
) -> None:
    _typed_experiment_command(
        ctx=ctx, spec_path=spec, expected=AttentionHeadRankSpec, events=events
    )


@attention_app.command("intervene")
def attention_intervene_command(
    ctx: typer.Context,
    spec: Annotated[
        str, typer.Option(help="Attention-intervention JSON/YAML spec.")
    ],
    events: Annotated[Literal["human", "jsonl", "none"], typer.Option()] = "none",
) -> None:
    _typed_experiment_command(
        ctx=ctx,
        spec_path=spec,
        expected=AttentionHeadInterventionSpec,
        events=events,
    )


def _execute_attention_trace_kind(
    *,
    ctx: typer.Context,
    spec_path: str,
    expected_trace_kind: Literal["token_edges", "head_paths"],
    events: Literal["human", "jsonl", "none"],
) -> None:
    try:
        parsed = load_spec(spec_path)
        if not isinstance(parsed, AttentionTraceSpec):
            raise SpecError(f"expected kind 'attention_trace', got {parsed.kind!r}")
        if parsed.trace_kind != expected_trace_kind:
            raise SpecError(
                f"expected trace_kind {expected_trace_kind!r}, got {parsed.trace_kind!r}"
            )
        context = _context(ctx)
        run_id = _execute_parsed(
            context=context,
            parsed=parsed,
            events=events,
            request_id=None,
            diagnostic_stream=sys.stderr,
        )
        _dump(
            _execution_receipt(context, run_id),
            mode="jsonl" if events == "jsonl" else "json",
        )
    except Exception as exc:
        _fail(exc, machine=True)


@attention_app.command("tokens")
def attention_tokens_command(
    ctx: typer.Context,
    spec: Annotated[str, typer.Option(help="Token-edge attention trace spec.")],
    events: Annotated[Literal["human", "jsonl", "none"], typer.Option()] = "none",
) -> None:
    _execute_attention_trace_kind(
        ctx=ctx,
        spec_path=spec,
        expected_trace_kind="token_edges",
        events=events,
    )


@attention_app.command("trace")
def attention_trace_command(
    ctx: typer.Context,
    spec: Annotated[str, typer.Option(help="Sender-to-receiver path trace spec.")],
    events: Annotated[Literal["human", "jsonl", "none"], typer.Option()] = "none",
) -> None:
    _execute_attention_trace_kind(
        ctx=ctx,
        spec_path=spec,
        expected_trace_kind="head_paths",
        events=events,
    )


@attention_app.command("heads")
def attention_heads_command(
    ctx: typer.Context,
    run_id: str,
    limit: Annotated[
        int, typer.Option("--top", "--limit", min=1, max=1000)
    ] = 50,
) -> None:
    try:
        summary = _summary(_context(ctx), run_id)
        heads = summary.get("heads")
        if not isinstance(heads, list):
            raise SpecError("run is not an attention-head ranking")
        _dump(
            {
                "schema_version": "probe.attention-head-query/v1",
                "run_id": run_id,
                "source_count": len(heads),
                "returned_count": min(limit, len(heads)),
                "heads": heads[:limit],
            }
        )
    except Exception as exc:
        _fail(exc, machine=True)


@attention_app.command("paths")
def attention_paths_command(
    ctx: typer.Context,
    run_id: str,
    limit: Annotated[int, typer.Option(min=1, max=1000)] = 50,
) -> None:
    try:
        summary = _summary(_context(ctx), run_id)
        paths = summary.get("paths")
        if not isinstance(paths, list):
            raise SpecError("run is not an attention head-path trace")
        ordered = sorted(
            paths,
            key=lambda item: -abs(float(item.get("path_specific_effect", 0.0))),
        )
        _dump(
            {
                "schema_version": "probe.attention-path-query/v1",
                "run_id": run_id,
                "source_count": len(paths),
                "returned_count": min(limit, len(paths)),
                "paths": ordered[:limit],
            }
        )
    except Exception as exc:
        _fail(exc, machine=True)


@app.command("workflow")
def workflow_command(
    ctx: typer.Context,
    driver: Annotated[
        str, typer.Option(help="Seeded workflow driver JSON/YAML file.")
    ],
    events: Annotated[Literal["human", "jsonl", "none"], typer.Option()] = "none",
) -> None:
    from .specs import load_document
    from .workflow import run_workflow

    try:
        context = _context(ctx)
        if context.endpoint:
            raise EndpointError(
                "workflow orchestration currently runs standalone; individual stages remain daemon-compatible"
            )
        parsed = ResearchWorkflowSpec.model_validate(load_document(driver))
        result = run_workflow(
            service=_service(context),
            spec=parsed,
            listener=_event_listener(events),
            diagnostic_stream=sys.stderr,
        )
        _dump(result, mode="jsonl" if events == "jsonl" else "json")
    except Exception as exc:
        _fail(exc, machine=True)


def _replay_run_data(
    context: CLIContext, run_id: str
) -> tuple[RankSpec, RunManifest | dict[str, Any], dict[str, Any], tuple[str, ...]]:
    if context.endpoint:
        from .client import ProbeClient

        client = ProbeClient.from_context(context)
        manifest = client.run_manifest(run_id)
        run_spec = RankSpec.model_validate(client.job_spec(manifest["job_id"]))
        summary = client.run_summary(run_id)
        verification = client.verify_run(run_id)
        return run_spec, manifest, summary, tuple(verification["failures"])
    repository = ArtifactRepository(context.workspace)
    manifest = repository.load_manifest(run_id)
    run_spec = repository.load_run_spec(run_id)
    if not isinstance(run_spec, RankSpec):
        raise SpecError("computational replay v1 supports rank runs only")
    return (
        run_spec,
        manifest,
        repository.load_summary(run_id),
        repository.verify(run_id),
    )


@replay_app.command("inspect")
def replay_inspect(driver: Path) -> None:
    from .replay import compact_replay_identity, load_replay_bundle

    try:
        bundle = load_replay_bundle(driver)
        _dump(compact_replay_identity(bundle))
    except Exception as exc:
        _fail(exc, machine=True)


@replay_app.command("record")
def replay_record(
    ctx: typer.Context,
    driver: Path,
    run_id: Annotated[str, typer.Option(help="Completed run to use as baseline.")],
    overwrite: Annotated[bool, typer.Option(help="Replace an existing baseline.")] = False,
) -> None:
    from .replay import load_replay_bundle, record_baseline

    try:
        bundle = load_replay_bundle(driver)
        run_spec, manifest, summary, failures = _replay_run_data(
            _context(ctx), run_id
        )
        baseline = record_baseline(
            bundle,
            run_spec=run_spec,
            manifest=manifest,
            summary=summary,
            integrity_failures=failures,
            overwrite=overwrite,
        )
        from .contracts import ReplayRecordReceipt

        _dump(
            ReplayRecordReceipt(
                driver_name=baseline.driver_name,
                baseline=str(bundle.baseline_path),
                source_run_id=baseline.source_run_id,
                science_hash=baseline.science_hash,
                request_hash=baseline.request_hash,
                ranking_top_n=bundle.driver.comparison.ranking_top_n,
            )
        )
    except Exception as exc:
        _fail(exc, machine=True)


@replay_app.command("check")
def replay_check(
    ctx: typer.Context,
    driver: Path,
    run_id: Annotated[str, typer.Option(help="Completed run to compare.")],
    output: Annotated[Literal["compact", "full"], typer.Option()] = "compact",
) -> None:
    from .replay import (
        compact_replay_report,
        compare_replay,
        load_baseline,
        load_replay_bundle,
        write_replay_report,
    )

    try:
        bundle = load_replay_bundle(driver)
        baseline = load_baseline(bundle)
        run_spec, manifest, summary, failures = _replay_run_data(
            _context(ctx), run_id
        )
        report = write_replay_report(
            bundle,
            compare_replay(
                bundle,
                baseline,
                run_spec=run_spec,
                manifest=manifest,
                summary=summary,
                integrity_failures=failures,
            ),
        )
        _dump(report if output == "full" else compact_replay_report(report))
        if report.verdict != "passed":
            raise typer.Exit(9)
    except typer.Exit:
        raise
    except Exception as exc:
        _fail(exc, machine=True)


@replay_app.command("run")
def replay_run(
    ctx: typer.Context,
    driver: Path,
    events: Annotated[Literal["human", "jsonl", "none"], typer.Option()] = "none",
    output: Annotated[Literal["compact", "full"], typer.Option()] = "compact",
    request_id: Annotated[
        str | None,
        typer.Option(help="Idempotency key; requires an explicit daemon endpoint."),
    ] = None,
) -> None:
    from .replay import (
        compact_replay_report,
        compare_replay,
        load_baseline,
        load_replay_bundle,
        write_replay_report,
    )

    try:
        bundle = load_replay_bundle(driver)
        baseline = load_baseline(bundle)
        context = _context(ctx)
        if request_id is not None and context.endpoint is None:
            raise SpecError("--request-id requires an explicit daemon endpoint")
        if request_id is not None:
            request_id = request_id.strip()
            if not request_id or len(request_id) > 256:
                raise SpecError(
                    "--request-id must contain 1 to 256 non-whitespace characters"
                )
        if context.endpoint:
            from .client import ProbeClient

            status = ProbeClient.from_context(context).run(
                bundle.spec,
                listener=_event_listener(events),
                request_id=request_id,
            )
            run_id = status["run_id"]
        else:
            outcome = _service(context).execute(
                bundle.spec,
                listener=_event_listener(events),
                diagnostic_stream=sys.stderr,
            )
            run_id = outcome.manifest.run_id
        run_spec, manifest, summary, failures = _replay_run_data(context, run_id)
        report = write_replay_report(
            bundle,
            compare_replay(
                bundle,
                baseline,
                run_spec=run_spec,
                manifest=manifest,
                summary=summary,
                integrity_failures=failures,
            ),
        )
        result = report if output == "full" else compact_replay_report(report)
        _dump(result, mode="jsonl" if events == "jsonl" else "json")
        if report.verdict != "passed":
            raise typer.Exit(9)
    except typer.Exit:
        raise
    except Exception as exc:
        _fail(exc, machine=True)


@jobs_app.command("status")
def jobs_status(ctx: typer.Context, job_id: str) -> None:
    try:
        context = _context(ctx)
        if context.endpoint:
            from .client import ProbeClient

            _dump(ProbeClient.from_context(context).job_status(job_id))
        else:
            _dump(ArtifactRepository(context.workspace).load_job(job_id))
    except Exception as exc:
        _fail(exc, machine=True)


@jobs_app.command("watch")
def jobs_watch(
    ctx: typer.Context,
    job_id: str,
    start_sequence: Annotated[int, typer.Option(min=0)] = 0,
) -> None:
    try:
        context = _context(ctx)
        if context.endpoint:
            from .client import ProbeClient

            ProbeClient.from_context(context).watch_job(
                job_id, start_sequence=start_sequence, listener=_event_listener("jsonl")
            )
        else:
            for event in ArtifactRepository(context.workspace).read_events(
                job_id, start_sequence=start_sequence
            ):
                _dump(event, mode="jsonl")
    except Exception as exc:
        _fail(exc, machine=True)


@jobs_app.command("spec")
def jobs_spec(ctx: typer.Context, job_id: str) -> None:
    try:
        context = _context(ctx)
        if context.endpoint:
            from .client import ProbeClient

            value = ProbeClient.from_context(context).job_spec(job_id)
        else:
            value = ArtifactRepository(context.workspace).load_job_spec(job_id)
        _dump(value)
    except Exception as exc:
        _fail(exc, machine=True)


@jobs_app.command("cancel")
def jobs_cancel(ctx: typer.Context, job_id: str) -> None:
    context = _context(ctx)
    if not context.endpoint:
        _fail(
            EndpointError("cancellation requires an explicit running daemon endpoint"),
            machine=True,
        )
    try:
        from .client import ProbeClient

        _dump(ProbeClient.from_context(context).cancel_job(job_id))
    except Exception as exc:
        _fail(exc, machine=True)


@runs_app.command("list")
def runs_list(ctx: typer.Context) -> None:
    try:
        context = _context(ctx)
        if context.endpoint:
            from .client import ProbeClient

            _dump(ProbeClient.from_context(context).runs())
        else:
            _dump([item.model_dump(mode="json") for item in ArtifactRepository(context.workspace).list_runs()])
    except Exception as exc:
        _fail(exc, machine=True)


@app.command("compare")
def compare_command(
    ctx: typer.Context,
    reference_run_id: str,
    candidate_run_ids: Annotated[list[str], typer.Argument(min=1)],
    top_n: Annotated[int, typer.Option(min=1)] = 50,
) -> None:
    from .comparison import compare_rank_runs
    from .contracts import RankRunSummary

    try:
        context = _context(ctx)
        reference_spec = _rank_run_spec(context, reference_run_id)
        reference = RankRunSummary.model_validate(
            _summary(context, reference_run_id)
        )
        candidate_specs = tuple(
            _rank_run_spec(context, run_id) for run_id in candidate_run_ids
        )
        candidates = tuple(
            (run_id, RankRunSummary.model_validate(_summary(context, run_id)))
            for run_id in candidate_run_ids
        )
        _dump(
            compare_rank_runs(
                reference_run_id=reference_run_id,
                reference=reference,
                candidates=candidates,
                top_n=top_n,
                reference_spec=reference_spec,
                candidate_specs=candidate_specs,
            )
        )
    except Exception as exc:
        _fail(exc, machine=True)


@app.command("stability")
def stability_command(
    ctx: typer.Context,
    run_id: str,
    top_n: Annotated[int, typer.Option(min=1)] = 50,
    splits: Annotated[int, typer.Option(min=1, max=1000)] = 100,
    bootstrap: Annotated[int, typer.Option(min=0, max=10000)] = 1000,
    seed: Annotated[int, typer.Option()] = 0,
) -> None:
    from safetensors.torch import load_file

    from .comparison import rank_stability
    from .contracts import RankRunSummary

    try:
        context = _context(ctx)
        if context.endpoint:
            raise EndpointError(
                "remote stability requires local tensor artifacts; export or run against the local workspace"
            )
        repository = ArtifactRepository(context.workspace)
        manifest = repository.load_manifest(run_id)
        if manifest.run_kind != "rank":
            raise SpecError("stability is defined for rank runs")
        summary = RankRunSummary.model_validate(repository.load_summary(run_id))
        tensors = load_file(repository.runs / run_id / "tensors.safetensors")
        _dump(
            rank_stability(
                run_id=run_id,
                summary=summary,
                tensors=tensors,
                top_n=top_n,
                splits=splits,
                bootstrap_iterations=bootstrap,
                seed=seed,
            )
        )
    except Exception as exc:
        _fail(exc, machine=True)


@app.command("claims")
def claims_command(ctx: typer.Context, run_id: str) -> None:
    try:
        summary = _summary(_context(ctx), run_id)
        _dump(
            {
                "schema_version": "probe.claims/v1",
                "run_id": run_id,
                "evidence_stage": summary.get("evidence_stage"),
                "claims": summary.get("claims", []),
                "warnings": summary.get("warnings", []),
            }
        )
    except Exception as exc:
        _fail(exc, machine=True)


@app.command("report")
def report_command(
    ctx: typer.Context,
    run_id: str,
    output: Annotated[
        Path | None,
        typer.Option(help="Output directory; defaults to WORKSPACE/reports/RUN_ID."),
    ] = None,
) -> None:
    from .reporting import build_research_report, write_research_report

    try:
        context = _context(ctx)
        if context.endpoint:
            from .client import ProbeClient

            client = ProbeClient.from_context(context)
            manifest = RunManifest.model_validate(client.run_manifest(run_id))
            summary = client.run_summary(run_id)
        else:
            repository = ArtifactRepository(context.workspace)
            failures = repository.verify(run_id)
            if failures:
                raise SpecError(
                    f"run integrity verification failed: {', '.join(failures)}"
                )
            manifest = repository.load_manifest(run_id)
            summary = repository.load_summary(run_id)
        report = build_research_report(
            run_id=run_id,
            manifest=manifest,
            summary=summary,
        )
        target = output or context.workspace / "reports" / run_id
        _dump(write_research_report(report=report, output_directory=target))
    except Exception as exc:
        _fail(exc, machine=True)


@app.command("sensitivity")
def sensitivity_command(
    ctx: typer.Context,
    run_id: str,
    metadata_key: Annotated[str, typer.Option()] = "perturbation_family",
    top_n: Annotated[int, typer.Option(min=1)] = 50,
) -> None:
    from safetensors.torch import load_file

    from .contracts import RankRunSummary
    from .sensitivity import perturbation_sensitivity

    try:
        context = _context(ctx)
        if context.endpoint:
            raise EndpointError(
                "remote sensitivity requires local tensor artifacts; export or use the local workspace"
            )
        repository = ArtifactRepository(context.workspace)
        spec = _rank_run_spec(context, run_id)
        summary = RankRunSummary.model_validate(repository.load_summary(run_id))
        tensors = load_file(repository.runs / run_id / "tensors.safetensors")
        _dump(
            perturbation_sensitivity(
                run_id=run_id,
                spec=spec,
                summary=summary,
                tensors=tensors,
                metadata_key=metadata_key,
                top_n=top_n,
            )
        )
    except Exception as exc:
        _fail(exc, machine=True)


@app.command("harvest")
def harvest_command(ctx: typer.Context, qualification_run_id: str) -> None:
    try:
        summary = _summary(_context(ctx), qualification_run_id)
        if summary.get("schema_version") != "probe.qualification-result/v1":
            raise SpecError("token harvesting requires a qualification run")
        counts: dict[tuple[str, str, int], dict[str, Any]] = {}
        for pair in summary["pairs"]:
            for item in pair["generated"]:
                token_ids = item.get("token_ids", [])
                if not token_ids:
                    continue
                key = (
                    item["condition"],
                    item["behavior_decision"],
                    int(token_ids[0]),
                )
                record = counts.setdefault(
                    key,
                    {
                        "condition": key[0],
                        "behavior_decision": key[1],
                        "token_id": key[2],
                        "count": 0,
                        "example_text": item["text"],
                        "pair_ids": [],
                    },
                )
                record["count"] += 1
                record["pair_ids"].append(pair["pair_id"])
        candidates = sorted(
            counts.values(),
            key=lambda item: (
                item["behavior_decision"],
                item["condition"],
                -item["count"],
                item["token_id"],
            ),
        )
        _dump(
            {
                "schema_version": "probe.token-harvest/v1",
                "qualification_run_id": qualification_run_id,
                "candidates": candidates,
                "warning": (
                    "Candidates are pilot observations; rerun ranking with predeclared token sets to test sensitivity."
                ),
            }
        )
    except Exception as exc:
        _fail(exc, machine=True)


def _summary(context: CLIContext, run_id: str) -> dict[str, Any]:
    if context.endpoint:
        from .client import ProbeClient

        return ProbeClient.from_context(context).run_summary(run_id)
    return ArtifactRepository(context.workspace).load_summary(run_id)


def _rank_run_spec(context: CLIContext, run_id: str) -> RankSpec:
    if context.endpoint:
        from .client import ProbeClient
        from .specs import parse_spec_data

        client = ProbeClient.from_context(context)
        manifest = client.run_manifest(run_id)
        parsed = parse_spec_data(client.job_spec(manifest["job_id"]))
    else:
        parsed = ArtifactRepository(context.workspace).load_run_spec(run_id)
    if not isinstance(parsed, RankSpec):
        raise SpecError(f"run {run_id!r} is not a rank run")
    return parsed


def _overview(context: CLIContext, run_id: str):
    if context.endpoint:
        from .client import ProbeClient

        return ProbeClient.from_context(context).run_overview(run_id)
    return build_overview(run_id=run_id, summary=_summary(context, run_id))


@runs_app.command("show")
def runs_show(ctx: typer.Context, run_id: str) -> None:
    try:
        _dump(_summary(_context(ctx), run_id))
    except Exception as exc:
        _fail(exc, machine=True)


@runs_app.command("overview")
def runs_overview(ctx: typer.Context, run_id: str) -> None:
    try:
        _dump(_overview(_context(ctx), run_id))
    except Exception as exc:
        _fail(exc, machine=True)


@runs_app.command("manifest")
def runs_manifest(ctx: typer.Context, run_id: str) -> None:
    try:
        context = _context(ctx)
        if context.endpoint:
            from .client import ProbeClient

            value = ProbeClient.from_context(context).run_manifest(run_id)
        else:
            value = ArtifactRepository(context.workspace).load_manifest(run_id)
        _dump(value)
    except Exception as exc:
        _fail(exc, machine=True)


@runs_app.command("layers")
def runs_layers(
    ctx: typer.Context,
    run_id: str,
    top: Annotated[int, typer.Option(min=1)] = 20,
    ranking_objective: Annotated[
        Literal["parent", "shared_direction", "effect_magnitude"], typer.Option()
    ] = "parent",
) -> None:
    try:
        summary = _summary(_context(ctx), run_id)
        objective = (
            str(summary.get("ranking_objective", "effect_magnitude"))
            if ranking_objective == "parent"
            else ranking_objective
        )
        values = list(summary["layers"])
        source_count = len(values)
        if objective == "shared_direction":
            if any(item.get("absolute_mean_mass") is None for item in values):
                raise ValueError("run does not contain shared-direction layer scores")
            values.sort(
                key=lambda item: (-float(item["absolute_mean_mass"]), int(item["layer"]))
            )
            sort = "absolute_mean_mass:desc,layer:asc"
        else:
            values.sort(
                key=lambda item: (-float(item["rms_mass"]), int(item["layer"]))
            )
            sort = "rms_mass:desc,layer:asc"
        selected = values[:top]
        _dump(
            query_envelope(
                run_id=run_id,
                query="layers",
                items=selected,
                source_count=source_count,
                matched_count=source_count,
                parameters={"top": top, "ranking_objective": objective},
                sort=sort,
            )
        )
    except Exception as exc:
        _fail(exc, machine=True)


@runs_app.command("neurons")
def runs_neurons(
    ctx: typer.Context,
    run_id: str,
    top: Annotated[int, typer.Option(min=1)] = 20,
    layer: Annotated[int | None, typer.Option(min=0)] = None,
    sign: Annotated[Literal["any", "positive", "negative"], typer.Option()] = "any",
    ranking_objective: Annotated[
        Literal["parent", "shared_direction", "effect_magnitude"], typer.Option()
    ] = "parent",
) -> None:
    try:
        summary = _summary(_context(ctx), run_id)
        objective = (
            str(summary.get("ranking_objective", "effect_magnitude"))
            if ranking_objective == "parent"
            else ranking_objective
        )
        view_key = (
            "shared_direction_neurons"
            if objective == "shared_direction"
            else "effect_magnitude_neurons"
        )
        values = list(summary.get(view_key) or [])
        if not values:
            if objective != summary.get("ranking_objective", "effect_magnitude"):
                raise ValueError(f"run does not contain {objective} candidates")
            values = list(summary["neurons"])
        source_count = len(values)
        if layer is not None:
            values = [item for item in values if item["layer"] == layer]
        if sign == "positive":
            values = [item for item in values if item["importance_mean"] > 0]
        elif sign == "negative":
            values = [item for item in values if item["importance_mean"] < 0]
        matched_count = len(values)
        selected = [
            {
                **item,
                "observable_effect": (
                    "toward_target"
                    if item["importance_mean"] > 0
                    else "toward_control"
                    if item["importance_mean"] < 0
                    else "neutral"
                ),
            }
            for item in values[:top]
        ]
        _dump(
            query_envelope(
                run_id=run_id,
                query="neurons",
                items=selected,
                source_count=source_count,
                matched_count=matched_count,
                parameters={
                    "top": top,
                    "layer": layer,
                    "sign": sign,
                    "ranking_objective": objective,
                },
                sort=(
                    "absolute_importance_mean:desc,rank:asc"
                    if objective == "shared_direction"
                    else "importance_rms:desc,rank:asc"
                ),
            )
        )
    except Exception as exc:
        _fail(exc, machine=True)


@runs_app.command("trajectory")
def runs_trajectory(
    ctx: typer.Context,
    run_id: str,
    pair: Annotated[str | None, typer.Option()] = None,
    metric: Annotated[
        Literal[
            "logit_gap",
            "target_probability",
            "target_rank",
            "entropy",
            "forward_kl",
            "paired_js",
            "total_variation",
        ],
        typer.Option(),
    ] = "logit_gap",
    checkpoint: Annotated[
        Literal["all", "block_input", "post_attention", "post_ffn"],
        typer.Option(),
    ] = "all",
    limit: Annotated[int, typer.Option(min=1, max=10_000)] = 500,
) -> None:
    from .contracts import TrajectoryRunSummary

    try:
        summary = TrajectoryRunSummary.model_validate(_summary(_context(ctx), run_id))
        source_count = sum(len(item.checkpoints) for item in summary.pairs)
        rows: list[dict[str, Any]] = []
        for pair_result in summary.pairs:
            if pair is not None and pair_result.pair_id != pair:
                continue
            for item in pair_result.checkpoints:
                if checkpoint != "all" and item.checkpoint != checkpoint:
                    continue
                values: dict[str, float | int]
                if metric == "logit_gap":
                    values = {
                        "original": item.original_gap,
                        "perturbed": item.perturbed_gap,
                        "paired_delta": item.pair_delta,
                    }
                elif metric == "target_probability":
                    values = {
                        "original": item.original_target_probability,
                        "perturbed": item.perturbed_target_probability,
                        "paired_delta": item.perturbed_target_probability
                        - item.original_target_probability,
                    }
                elif metric == "target_rank":
                    values = {
                        "original": item.original_target_rank,
                        "perturbed": item.perturbed_target_rank,
                        "paired_delta": item.perturbed_target_rank
                        - item.original_target_rank,
                    }
                elif metric == "entropy":
                    values = {
                        "original": item.original_entropy,
                        "perturbed": item.perturbed_entropy,
                        "paired_delta": item.perturbed_entropy
                        - item.original_entropy,
                    }
                elif metric == "forward_kl":
                    values = {
                        "original": item.original_forward_kl_to_final,
                        "perturbed": item.perturbed_forward_kl_to_final,
                        "paired_delta": item.perturbed_forward_kl_to_final
                        - item.original_forward_kl_to_final,
                    }
                elif metric == "paired_js":
                    values = {"paired": item.paired_js}
                else:
                    values = {"paired": item.paired_total_variation}
                rows.append(
                    {
                        "pair_id": pair_result.pair_id,
                        "split": pair_result.split,
                        "layer": item.layer,
                        "checkpoint": item.checkpoint,
                        **values,
                    }
                )
        matched_count = len(rows)
        _dump(
            query_envelope(
                run_id=run_id,
                query="trajectory",
                items=rows[:limit],
                source_count=source_count,
                matched_count=matched_count,
                parameters={
                    "pair": pair,
                    "metric": metric,
                    "checkpoint": checkpoint,
                    "limit": limit,
                    "lower_is_better": metric == "target_rank",
                },
                sort="pair:parent,layer:asc,checkpoint:architecture",
            )
        )
    except Exception as exc:
        _fail(exc, machine=True)


@runs_app.command("trajectory-visualize")
def runs_trajectory_visualize(
    ctx: typer.Context,
    trajectory_run_id: str,
    intervention_run_ids: Annotated[
        list[str], typer.Argument(min=1, help="One or more intervention run IDs.")
    ],
    output: Annotated[Path, typer.Option(help="Self-contained HTML output path.")],
    pair: Annotated[str | None, typer.Option(help="Pair ID; defaults to the first pair.")] = None,
) -> None:
    from hashlib import sha256

    from .contracts import InterventionRunSummary, TrajectoryRunSummary
    from .trajectory_visualization import (
        render_trajectory_visualization,
        write_trajectory_visualization,
    )

    try:
        context = _context(ctx)
        run_ids = [trajectory_run_id, *intervention_run_ids]
        verification: dict[str, bool] = {}
        for run_id in run_ids:
            if context.endpoint:
                from .client import ProbeClient

                report = ProbeClient.from_context(context).verify_run(run_id)
                failures = tuple(report["failures"])
            else:
                failures = ArtifactRepository(context.workspace).verify(run_id)
            verification[run_id] = not failures
            if failures:
                raise ArtifactError(
                    f"run {run_id!r} failed artifact verification: {', '.join(failures)}"
                )
        trajectory = TrajectoryRunSummary.model_validate(
            _summary(context, trajectory_run_id)
        )
        interventions = tuple(
            (
                run_id,
                InterventionRunSummary.model_validate(_summary(context, run_id)),
            )
            for run_id in intervention_run_ids
        )
        html = render_trajectory_visualization(
            trajectory_run_id=trajectory_run_id,
            trajectory=trajectory,
            intervention_runs=interventions,
            pair_id=pair,
        )
        destination = write_trajectory_visualization(output, html)
        data = destination.read_bytes()
        selected_pair = pair or trajectory.pairs[0].pair_id
        _dump(
            {
                "schema_version": "probe.trajectory-visualization-receipt/v1",
                "trajectory_run_id": trajectory_run_id,
                "intervention_run_ids": intervention_run_ids,
                "pair_id": selected_pair,
                "verified_runs": verification,
                "output": str(destination),
                "sha256": sha256(data).hexdigest(),
                "size_bytes": len(data),
            }
        )
    except Exception as exc:
        _fail(exc, machine=True)


@runs_app.command("transitions")
def runs_transitions(
    ctx: typer.Context,
    run_id: str,
    pair: Annotated[str | None, typer.Option()] = None,
    split: Annotated[
        Literal["all", "discovery", "validation", "heldout"], typer.Option()
    ] = "all",
    limit: Annotated[int, typer.Option(min=1, max=1000)] = 20,
) -> None:
    from .contracts import TrajectoryRunSummary

    try:
        summary = TrajectoryRunSummary.model_validate(_summary(_context(ctx), run_id))
        source_count = sum(len(item.transitions) for item in summary.pairs)
        rows = [
            {
                "pair_id": pair_result.pair_id,
                "split": pair_result.split,
                **transition.model_dump(mode="json"),
            }
            for pair_result in summary.pairs
            if (pair is None or pair_result.pair_id == pair)
            and (split == "all" or pair_result.split == split)
            for transition in pair_result.transitions
        ]
        rows.sort(
            key=lambda item: (
                -float(item["absolute_change"]),
                str(item["pair_id"]),
                int(item["layer"]),
            )
        )
        matched_count = len(rows)
        _dump(
            query_envelope(
                run_id=run_id,
                query="transitions",
                items=rows[:limit],
                source_count=source_count,
                matched_count=matched_count,
                parameters={"pair": pair, "split": split, "limit": limit},
                sort="absolute_change:desc,pair_id:asc,layer:asc",
            )
        )
    except Exception as exc:
        _fail(exc, machine=True)


@runs_app.command("ffn-couplings")
def runs_ffn_couplings(
    ctx: typer.Context,
    run_id: str,
    method: Annotated[
        Literal["direct", "native", "downstream"], typer.Option()
    ] = "downstream",
    top: Annotated[int, typer.Option(min=1, max=10_000)] = 20,
    layer: Annotated[int | None, typer.Option(min=0)] = None,
    sign: Annotated[
        Literal["any", "positive", "negative"], typer.Option()
    ] = "any",
    ranking_objective: Annotated[
        Literal["parent", "shared_direction", "effect_magnitude"], typer.Option()
    ] = "parent",
) -> None:
    from .contracts import FFNCouplingRunSummary

    try:
        summary = FFNCouplingRunSummary.model_validate(_summary(_context(ctx), run_id))
        objective = (
            summary.ranking_objective
            if ranking_objective == "parent"
            else ranking_objective
        )
        candidates = (
            summary.shared_direction_neurons
            if objective == "shared_direction"
            else summary.effect_magnitude_neurons
        )
        if not candidates:
            if objective != summary.ranking_objective:
                raise ValueError(f"run does not contain {objective} candidates")
            candidates = summary.neurons
        source_count = len(candidates)
        rows: list[dict[str, Any]] = []
        for item in candidates:
            if layer is not None and item.layer != layer:
                continue
            if method == "direct":
                coupling = item.direct_coupling
                importance = item.direct_importance_mean
                rms = item.direct_importance_rms
            elif method == "native":
                if (
                    item.native_coupling_mean is None
                    or item.native_importance_mean is None
                    or item.native_importance_rms is None
                ):
                    continue
                coupling = item.native_coupling_mean
                importance = item.native_importance_mean
                rms = item.native_importance_rms
            else:
                coupling = item.downstream_coupling_mean
                importance = item.downstream_importance_mean
                rms = item.downstream_importance_rms
            if sign == "positive" and importance <= 0:
                continue
            if sign == "negative" and importance >= 0:
                continue
            rows.append(
                {
                    "rank": item.rank,
                    "layer": item.layer,
                    "neuron": item.neuron,
                    "method": method,
                    "activation_delta_mean": item.activation_delta_mean,
                    "coupling": coupling,
                    "importance_mean": importance,
                    "importance_rms": rms,
                    "downstream_sign_consistency": item.downstream_sign_consistency,
                    "direct_downstream_sign_agreement": item.direct_downstream_sign_agreement,
                }
            )
        rows.sort(
            key=lambda item: (
                -(
                    abs(float(item["importance_mean"]))
                    if objective == "shared_direction"
                    else float(item["importance_rms"])
                ),
                int(item["layer"]),
                int(item["neuron"]),
            )
        )
        matched_count = len(rows)
        _dump(
            query_envelope(
                run_id=run_id,
                query="ffn_couplings",
                items=rows[:top],
                source_count=source_count,
                matched_count=matched_count,
                parameters={
                    "method": method,
                    "top": top,
                    "layer": layer,
                    "sign": sign,
                    "candidate_pair_ids": list(summary.candidate_pair_ids),
                    "ranking_objective": objective,
                },
                sort=(
                    "absolute_importance_mean:desc,layer:asc,neuron:asc"
                    if objective == "shared_direction"
                    else "importance_rms:desc,layer:asc,neuron:asc"
                ),
            )
        )
    except Exception as exc:
        _fail(exc, machine=True)


@runs_app.command("coupling-compare")
def runs_coupling_compare(
    ctx: typer.Context,
    run_id: str,
    top: Annotated[int, typer.Option(min=1, max=10_000)] = 50,
    layer: Annotated[int | None, typer.Option(min=0)] = None,
) -> None:
    from .contracts import FFNCouplingRunSummary

    try:
        summary = FFNCouplingRunSummary.model_validate(_summary(_context(ctx), run_id))
        source_count = len(summary.neurons)
        epsilon = 1e-12
        rows = [
            {
                "rank": item.rank,
                "layer": item.layer,
                "neuron": item.neuron,
                "direct_importance_rms": item.direct_importance_rms,
                "native_importance_rms": item.native_importance_rms,
                "downstream_importance_rms": item.downstream_importance_rms,
                "downstream_to_direct_ratio": (
                    item.downstream_importance_rms + epsilon
                )
                / (item.direct_importance_rms + epsilon),
                "log10_downstream_to_direct": math.log10(
                    (item.downstream_importance_rms + epsilon)
                    / (item.direct_importance_rms + epsilon)
                ),
                "direct_downstream_sign_agreement": item.direct_downstream_sign_agreement,
            }
            for item in summary.neurons
            if layer is None or item.layer == layer
        ]
        rows.sort(
            key=lambda item: (
                -abs(float(item["log10_downstream_to_direct"])),
                int(item["layer"]),
                int(item["neuron"]),
            )
        )
        matched_count = len(rows)
        _dump(
            query_envelope(
                run_id=run_id,
                query="coupling_compare",
                items=rows[:top],
                source_count=source_count,
                matched_count=matched_count,
                parameters={
                    "top": top,
                    "layer": layer,
                    "candidate_pair_ids": list(summary.candidate_pair_ids),
                },
                sort="abs(log10_downstream_to_direct):desc,layer:asc,neuron:asc",
            )
        )
    except Exception as exc:
        _fail(exc, machine=True)


@runs_app.command("files")
def runs_files(ctx: typer.Context, run_id: str) -> None:
    try:
        context = _context(ctx)
        if context.endpoint:
            from .client import ProbeClient

            values = ProbeClient.from_context(context).run_manifest(run_id)["artifacts"]
        else:
            values = ArtifactRepository(context.workspace).load_manifest(run_id).artifacts
        items = sorted(
            JSONABLE_ADAPTER.dump_python(values, mode="json"),
            key=lambda item: item["path"],
        )
        _dump(
            query_envelope(
                run_id=run_id,
                query="files",
                items=items,
                source_count=len(items),
                sort="path:asc",
            )
        )
    except Exception as exc:
        _fail(exc, machine=True)


@runs_app.command("verify")
def runs_verify(ctx: typer.Context, run_id: str) -> None:
    from .contracts import VerificationReport

    try:
        context = _context(ctx)
        if context.endpoint:
            from .client import ProbeClient

            result = ProbeClient.from_context(context).verify_run(run_id)
        else:
            failures = ArtifactRepository(context.workspace).verify(run_id)
            result = VerificationReport(
                run_id=run_id,
                valid=not failures,
                failures=tuple(failures),
            ).model_dump(mode="json")
        _dump(result)
        if not result["valid"]:
            raise typer.Exit(6)
    except typer.Exit:
        raise
    except Exception as exc:
        _fail(exc, machine=True)


@runs_app.command("export")
def runs_export(ctx: typer.Context, run_id: str, output: Path) -> None:
    try:
        context = _context(ctx)
        if context.endpoint:
            raise EndpointError("remote export is not implemented; download listed artifacts")
        repository = ArtifactRepository(context.workspace)
        repository.load_manifest(run_id)
        source = repository.runs / run_id
        base = output.with_suffix("")
        archive = shutil.make_archive(str(base), "gztar", root_dir=source)
        _dump({"run_id": run_id, "archive": archive})
    except Exception as exc:
        _fail(exc, machine=True)


@app.command()
def tui(
    model: Annotated[str, typer.Option()] = DEFAULT_MODEL,
    revision: Annotated[str | None, typer.Option()] = None,
    device: Annotated[str, typer.Option()] = "auto",
    dtype: Annotated[str, typer.Option()] = "auto",
    sample: Annotated[str, typer.Option()] = "agreement-capital",
    top_k: Annotated[int, typer.Option(min=1)] = 500,
    local_files_only: Annotated[bool, typer.Option()] = True,
) -> None:
    arguments = [
        "--model", model,
        "--device", device,
        "--dtype", dtype,
        "--sample", sample,
        "--top-k", str(top_k),
    ]
    if revision:
        arguments.extend(["--revision", revision])
    if local_files_only:
        arguments.append("--local-files-only")
    legacy_main(arguments)


@app.command()
def serve(
    ctx: typer.Context,
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8765,
) -> None:
    from .server import serve as run_server

    context = _context(ctx)
    try:
        run_server(
            workspace=context.workspace,
            cache_dir=context.cache_dir,
            host=host,
            port=port,
        )
    except Exception as exc:
        _fail(exc)


@server_app.command("start")
def server_start(
    ctx: typer.Context,
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8765,
) -> None:
    from .server_process import start_server

    context = _context(ctx)
    try:
        _dump(start_server(context.workspace, context.cache_dir, port=port))
    except Exception as exc:
        _fail(exc, machine=True)


@server_app.command("status")
def server_status(ctx: typer.Context) -> None:
    from .server_process import server_status as status

    try:
        _dump(status(_context(ctx).workspace))
    except Exception as exc:
        _fail(exc, machine=True)


@server_app.command("stop")
def server_stop(ctx: typer.Context) -> None:
    from .server_process import stop_server

    try:
        _dump(stop_server(_context(ctx).workspace))
    except Exception as exc:
        _fail(exc, machine=True)


def main(argv: list[str] | None = None) -> None:
    app(args=argv, prog_name="probe")


__all__ = ["app", "legacy_main", "main"]
