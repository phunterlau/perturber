from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    RankRunSummary,
    RankSpec,
    ReplayBaseline,
    ReplayCheck,
    ReplayDriver,
    ReplayIdentity,
    ReplayOutcome,
    ReplayReport,
    RunManifest,
)
from .errors import SpecError
from .specs import load_document, load_spec, request_hash, science_hash


STABLE_REPLAY_ARTIFACTS = (
    "spec.json",
    "layers.csv",
    "neurons.csv",
    "tensors.safetensors",
)


@dataclass(frozen=True)
class ReplayBundle:
    driver_path: Path
    root: Path
    driver: ReplayDriver
    spec_path: Path
    baseline_path: Path
    report_directory: Path
    spec: RankSpec


def _bundle_path(root: Path, relative: str) -> Path:
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise SpecError(f"replay path escapes bundle directory: {relative!r}")
    return resolved


def load_replay_bundle(driver_path: str | Path) -> ReplayBundle:
    path = Path(driver_path).resolve()
    if not path.is_file():
        raise SpecError(f"replay driver was not found: {path}")
    try:
        driver = ReplayDriver.model_validate(load_document(path))
    except (OSError, ValueError) as exc:
        raise SpecError(f"invalid replay driver {path}: {exc}") from exc
    root = path.parent.resolve()
    spec_path = _bundle_path(root, driver.spec)
    baseline_path = _bundle_path(root, driver.baseline)
    report_directory = _bundle_path(root, driver.report_directory)
    try:
        spec = load_spec(spec_path)
    except OSError as exc:
        raise SpecError(f"could not load replay spec {spec_path}: {exc}") from exc
    if not isinstance(spec, RankSpec):
        raise SpecError("computational replay v1 supports rank specs only")
    _validate_driver_spec(driver, spec)
    return ReplayBundle(
        driver_path=path,
        root=root,
        driver=driver,
        spec_path=spec_path,
        baseline_path=baseline_path,
        report_directory=report_directory,
        spec=spec,
    )


def _validate_driver_spec(driver: ReplayDriver, spec: RankSpec) -> None:
    record = driver.reproducibility
    mismatches: list[str] = []
    expected = {
        "execution.seed": (record.torch_seed, spec.execution.seed),
        "model.revision": (record.resolved_model_revision, spec.model.revision),
        "model.device": (record.device, spec.model.device),
        "model.dtype": (record.model_dtype, spec.model.dtype),
    }
    if spec.model.adapter != "auto" and not record.adapter.startswith(spec.model.adapter):
        mismatches.append(
            "model.adapter "
            f"driver={record.adapter!r} is incompatible with spec={spec.model.adapter!r}"
        )
    for field, (driver_value, spec_value) in expected.items():
        if driver_value != spec_value:
            mismatches.append(
                f"{field} driver={driver_value!r} spec={spec_value!r}"
            )
    if mismatches:
        raise SpecError(
            "replay driver and spec disagree: " + "; ".join(mismatches)
        )


def load_baseline(bundle: ReplayBundle) -> ReplayBaseline:
    if not bundle.baseline_path.is_file():
        raise SpecError(
            f"replay baseline was not found: {bundle.baseline_path}; "
            "record one with 'probe replay record'"
        )
    try:
        return ReplayBaseline.model_validate_json(
            bundle.baseline_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise SpecError(f"invalid replay baseline {bundle.baseline_path}: {exc}") from exc


def _as_manifest(value: RunManifest | Mapping[str, Any]) -> RunManifest:
    return value if isinstance(value, RunManifest) else RunManifest.model_validate(value)


def _as_summary(value: RankRunSummary | Mapping[str, Any]) -> RankRunSummary:
    return value if isinstance(value, RankRunSummary) else RankRunSummary.model_validate(value)


def _artifact_hashes(manifest: RunManifest) -> dict[str, str]:
    available = {item.path: item.sha256 for item in manifest.artifacts}
    return {
        path: available[path]
        for path in STABLE_REPLAY_ARTIFACTS
        if path in available
    }


def _driver_run_mismatches(
    bundle: ReplayBundle,
    run_spec: RankSpec,
    manifest: RunManifest,
) -> list[str]:
    record = bundle.driver.reproducibility
    model = manifest.resolved_model
    mismatches: list[str] = []
    expected = {
        "science_hash": (science_hash(bundle.spec), manifest.science_hash),
        "request_hash": (request_hash(bundle.spec), request_hash(run_spec)),
        "requested revision": (
            record.resolved_model_revision,
            run_spec.model.revision,
        ),
        "resolved revision": (
            record.resolved_model_revision,
            model.get("resolved_revision"),
        ),
        "adapter": (record.adapter, model.get("adapter")),
        "device": (record.device, model.get("device")),
        "model dtype": (record.model_dtype, model.get("dtype")),
        "torch seed": (record.torch_seed, run_spec.execution.seed),
    }
    for field, (expected_value, actual_value) in expected.items():
        if expected_value != actual_value:
            mismatches.append(
                f"{field}: expected {expected_value!r}, got {actual_value!r}"
            )
    for key, expected_value in record.expected_environment.items():
        actual_value = manifest.environment.get(key)
        if actual_value != expected_value:
            mismatches.append(
                f"environment.{key}: expected {expected_value!r}, got {actual_value!r}"
            )
    return mismatches


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value.model_dump(mode="json") if hasattr(value, "model_dump") else value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def record_baseline(
    bundle: ReplayBundle,
    *,
    run_spec: RankSpec,
    manifest: RunManifest | Mapping[str, Any],
    summary: RankRunSummary | Mapping[str, Any],
    integrity_failures: tuple[str, ...] = (),
    overwrite: bool = False,
) -> ReplayBaseline:
    if bundle.baseline_path.exists() and not overwrite:
        raise SpecError(
            f"replay baseline already exists: {bundle.baseline_path}; use --overwrite"
        )
    manifest_value = _as_manifest(manifest)
    summary_value = _as_summary(summary)
    mismatches = _driver_run_mismatches(bundle, run_spec, manifest_value)
    if integrity_failures:
        mismatches.append("artifact integrity: " + ", ".join(integrity_failures))
    if summary_value.science_hash != manifest_value.science_hash:
        mismatches.append("summary science_hash differs from manifest")
    if mismatches:
        raise SpecError("run cannot be recorded as this baseline: " + "; ".join(mismatches))

    summary_data = summary_value.model_dump(mode="json")
    summary_data["neurons"] = summary_data["neurons"][
        : bundle.driver.comparison.ranking_top_n
    ]
    baseline = ReplayBaseline(
        driver_name=bundle.driver.name,
        recorded_at=datetime.now(timezone.utc),
        source_run_id=manifest_value.run_id,
        science_hash=manifest_value.science_hash,
        request_hash=request_hash(run_spec),
        run_fingerprint=manifest_value.run_fingerprint,
        algorithm_version=manifest_value.algorithm_version,
        resolved_model=manifest_value.resolved_model,
        environment=manifest_value.environment,
        artifact_hashes=_artifact_hashes(manifest_value),
        summary=summary_data,
    )
    _write_json_atomic(bundle.baseline_path, baseline)
    return baseline


def _check(
    checks: list[ReplayCheck],
    name: str,
    baseline: Any,
    replay: Any,
    *,
    required: bool = True,
    detail: str | None = None,
) -> None:
    checks.append(
        ReplayCheck(
            name=name,
            passed=baseline == replay,
            required=required,
            baseline=baseline,
            replay=replay,
            detail=detail,
        )
    )


def _numeric_values(summary: Mapping[str, Any]) -> dict[str, float | None]:
    values: dict[str, float | None] = {
        "aggregate.measured_delta_mean": summary.get("measured_delta_mean"),
        "aggregate.predicted_delta_mean": summary.get("predicted_delta_mean"),
        "aggregate.ffn_skip_mean": summary.get("ffn_skip_mean"),
    }
    for pair in summary.get("pairs", []):
        prefix = f"pair.{pair['pair_id']}"
        for field in (
            "original_gap",
            "perturbed_gap",
            "measured_delta",
            "predicted_delta",
            "ffn_skip_mean",
        ):
            values[f"{prefix}.{field}"] = pair.get(field)
    for layer in summary.get("layers", []):
        prefix = f"layer.{layer['layer']}"
        for field in (
            "signed_mean_sum",
            "rms_mass",
            "activation_delta_norm_mean",
        ):
            values[f"{prefix}.{field}"] = layer.get(field)
    return values


def _neuron_numeric_values(
    summary: Mapping[str, Any], top_n: int
) -> dict[str, float]:
    values: dict[str, float] = {}
    for neuron in summary.get("neurons", [])[:top_n]:
        prefix = f"neuron.{neuron['layer']}:{neuron['neuron']}"
        values[f"{prefix}.importance_mean"] = float(neuron["importance_mean"])
        values[f"{prefix}.importance_rms"] = float(neuron["importance_rms"])
    return values


def _prediction_values(summary: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    return {
        pair["pair_id"]: (
            pair["original_prediction"],
            pair["perturbed_prediction"],
        )
        for pair in summary.get("pairs", [])
    }


def _neuron_key(value: Mapping[str, Any]) -> tuple[int, int]:
    return int(value["layer"]), int(value["neuron"])


def _sign(value: float) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def compare_replay(
    bundle: ReplayBundle,
    baseline: ReplayBaseline,
    *,
    run_spec: RankSpec,
    manifest: RunManifest | Mapping[str, Any],
    summary: RankRunSummary | Mapping[str, Any],
    integrity_failures: tuple[str, ...] = (),
) -> ReplayReport:
    manifest_value = _as_manifest(manifest)
    summary_value = _as_summary(summary)
    replay_summary = summary_value.model_dump(mode="json")
    baseline_summary = baseline.summary
    policy = bundle.driver.comparison
    checks: list[ReplayCheck] = []

    _check(checks, "driver_name", baseline.driver_name, bundle.driver.name)
    _check(checks, "artifact_integrity", [], list(integrity_failures))
    _check(checks, "science_hash", baseline.science_hash, manifest_value.science_hash)
    _check(checks, "request_hash", baseline.request_hash, request_hash(run_spec))
    _check(
        checks,
        "driver_science_hash",
        science_hash(bundle.spec),
        manifest_value.science_hash,
    )
    _check(checks, "run_fingerprint", baseline.run_fingerprint, manifest_value.run_fingerprint)
    _check(checks, "algorithm_version", baseline.algorithm_version, manifest_value.algorithm_version)
    _check(checks, "resolved_model", baseline.resolved_model, manifest_value.resolved_model)
    _check(
        checks,
        "observable",
        baseline_summary.get("observable"),
        replay_summary.get("observable"),
    )
    _check(
        checks,
        "pair_ids",
        [item["pair_id"] for item in baseline_summary.get("pairs", [])],
        [item["pair_id"] for item in replay_summary.get("pairs", [])],
    )

    driver_mismatches = _driver_run_mismatches(bundle, run_spec, manifest_value)
    _check(
        checks,
        "driver_execution_contract",
        [],
        driver_mismatches,
        detail="Pinned revision, adapter, device, dtype, seed, and selected environment.",
    )
    expected_environment = bundle.driver.reproducibility.expected_environment
    _check(
        checks,
        "baseline_environment_contract",
        expected_environment,
        {key: baseline.environment.get(key) for key in expected_environment},
    )

    baseline_numbers = _numeric_values(baseline_summary)
    replay_numbers = _numeric_values(replay_summary)
    numeric_rows: list[dict[str, Any]] = []
    for name in sorted(set(baseline_numbers) | set(replay_numbers)):
        expected_value = baseline_numbers.get(name)
        actual_value = replay_numbers.get(name)
        if expected_value is None or actual_value is None:
            passed = expected_value is actual_value
            absolute_difference = None
        else:
            absolute_difference = abs(float(actual_value) - float(expected_value))
            passed = math.isclose(
                float(actual_value),
                float(expected_value),
                rel_tol=policy.scalar_relative_tolerance,
                abs_tol=policy.scalar_absolute_tolerance,
            )
        numeric_rows.append(
            {
                "name": name,
                "baseline": expected_value,
                "replay": actual_value,
                "absolute_difference": absolute_difference,
                "passed": passed,
            }
        )
    numeric_passed = all(row["passed"] for row in numeric_rows)
    checks.append(
        ReplayCheck(
            name="numeric_tolerances",
            passed=numeric_passed,
            baseline={
                "absolute": policy.scalar_absolute_tolerance,
                "relative": policy.scalar_relative_tolerance,
            },
            replay={
                "passed": sum(bool(row["passed"]) for row in numeric_rows),
                "total": len(numeric_rows),
            },
        )
    )

    if policy.require_exact_predictions:
        _check(
            checks,
            "predictions",
            _prediction_values(baseline_summary),
            _prediction_values(replay_summary),
        )

    limit = policy.ranking_top_n
    baseline_neurons = baseline_summary.get("neurons", [])[:limit]
    replay_neurons = replay_summary.get("neurons", [])[:limit]
    baseline_by_key = {_neuron_key(item): item for item in baseline_neurons}
    replay_by_key = {_neuron_key(item): item for item in replay_neurons}
    overlap_keys = set(baseline_by_key) & set(replay_by_key)
    denominator = min(limit, len(baseline_by_key), len(replay_by_key))
    overlap = len(overlap_keys) / denominator if denominator else 1.0
    sign_agreement = (
        sum(
            _sign(float(baseline_by_key[key]["importance_mean"]))
            == _sign(float(replay_by_key[key]["importance_mean"]))
            for key in overlap_keys
        )
        / len(overlap_keys)
        if overlap_keys
        else 1.0
    )
    rank_displacements = [
        abs(int(baseline_by_key[key]["rank"]) - int(replay_by_key[key]["rank"]))
        for key in overlap_keys
    ]
    mean_rank_displacement = (
        sum(rank_displacements) / len(rank_displacements)
        if rank_displacements
        else 0.0
    )
    checks.append(
        ReplayCheck(
            name="top_neuron_overlap",
            passed=overlap >= policy.minimum_top_n_overlap,
            baseline=policy.minimum_top_n_overlap,
            replay=overlap,
        )
    )
    checks.append(
        ReplayCheck(
            name="mean_rank_displacement",
            passed=(
                mean_rank_displacement <= policy.maximum_mean_rank_displacement
            ),
            baseline=policy.maximum_mean_rank_displacement,
            replay=mean_rank_displacement,
        )
    )

    baseline_neuron_numbers = _neuron_numeric_values(baseline_summary, limit)
    replay_neuron_numbers = _neuron_numeric_values(replay_summary, limit)
    neuron_metric_rows: list[dict[str, Any]] = []
    for name in sorted(set(baseline_neuron_numbers) | set(replay_neuron_numbers)):
        expected_value = baseline_neuron_numbers.get(name)
        actual_value = replay_neuron_numbers.get(name)
        passed = (
            expected_value is not None
            and actual_value is not None
            and math.isclose(
                actual_value,
                expected_value,
                rel_tol=policy.scalar_relative_tolerance,
                abs_tol=policy.scalar_absolute_tolerance,
            )
        )
        neuron_metric_rows.append(
            {
                "name": name,
                "baseline": expected_value,
                "replay": actual_value,
                "absolute_difference": (
                    abs(actual_value - expected_value)
                    if expected_value is not None and actual_value is not None
                    else None
                ),
                "passed": passed,
            }
        )
    neuron_metrics_passed = all(row["passed"] for row in neuron_metric_rows)
    maximum_neuron_absolute_difference = max(
        (
            row["absolute_difference"]
            for row in neuron_metric_rows
            if row["absolute_difference"] is not None
        ),
        default=0.0,
    )
    checks.append(
        ReplayCheck(
            name="top_neuron_importance_tolerances",
            passed=neuron_metrics_passed,
            baseline={
                "absolute": policy.scalar_absolute_tolerance,
                "relative": policy.scalar_relative_tolerance,
            },
            replay={
                "passed": sum(bool(row["passed"]) for row in neuron_metric_rows),
                "total": len(neuron_metric_rows),
            },
        )
    )
    checks.append(
        ReplayCheck(
            name="neuron_sign_agreement",
            passed=sign_agreement >= policy.minimum_sign_agreement,
            baseline=policy.minimum_sign_agreement,
            replay=sign_agreement,
        )
    )

    replay_artifacts = _artifact_hashes(manifest_value)
    hash_rows = {
        path: {
            "baseline": baseline.artifact_hashes.get(path),
            "replay": replay_artifacts.get(path),
            "matched": baseline.artifact_hashes.get(path) == replay_artifacts.get(path),
        }
        for path in sorted(set(baseline.artifact_hashes) | set(replay_artifacts))
    }
    all_hashes_match = bool(hash_rows) and all(
        value["matched"] for value in hash_rows.values()
    )
    checks.append(
        ReplayCheck(
            name="stable_artifact_hashes",
            passed=all_hashes_match,
            required=policy.artifact_hashes == "require",
            baseline=baseline.artifact_hashes,
            replay=replay_artifacts,
            detail=(
                "Stable scientific artifacts only; manifests, events, summaries, "
                "run IDs, and timing are excluded."
            ),
        )
    )

    required_passed = all(item.passed for item in checks if item.required)
    warnings = []
    if not all_hashes_match and policy.artifact_hashes == "report":
        warnings.append(
            "Stable artifact bytes differed, but hashes are report-only under this policy."
        )
    return ReplayReport(
        driver_name=bundle.driver.name,
        baseline_run_id=baseline.source_run_id,
        replay_run_id=manifest_value.run_id,
        created_at=datetime.now(timezone.utc),
        verdict="passed" if required_passed else "failed",
        checks=tuple(checks),
        ranking={
            "top_n": denominator,
            "overlap_count": len(overlap_keys),
            "overlap_fraction": overlap,
            "minimum_overlap": policy.minimum_top_n_overlap,
            "sign_agreement": sign_agreement,
            "minimum_sign_agreement": policy.minimum_sign_agreement,
            "mean_rank_displacement": mean_rank_displacement,
            "maximum_mean_rank_displacement": policy.maximum_mean_rank_displacement,
            "neuron_metrics_passed": sum(
                bool(row["passed"]) for row in neuron_metric_rows
            ),
            "neuron_metrics_total": len(neuron_metric_rows),
            "maximum_neuron_absolute_difference": maximum_neuron_absolute_difference,
            "neuron_metrics": neuron_metric_rows,
        },
        numeric={
            "absolute_tolerance": policy.scalar_absolute_tolerance,
            "relative_tolerance": policy.scalar_relative_tolerance,
            "passed_count": sum(bool(row["passed"]) for row in numeric_rows),
            "total_count": len(numeric_rows),
            "maximum_absolute_difference": max(
                (
                    row["absolute_difference"]
                    for row in numeric_rows
                    if row["absolute_difference"] is not None
                ),
                default=0.0,
            ),
            "metrics": numeric_rows,
        },
        artifact_hashes={
            "policy": policy.artifact_hashes,
            "all_matched": all_hashes_match,
            "artifacts": hash_rows,
        },
        warnings=tuple(warnings),
    )


def _markdown_report(report: ReplayReport) -> str:
    status = "PASS" if report.verdict == "passed" else "FAIL"
    required = [item for item in report.checks if item.required]
    lines = [
        f"# Replay report: {report.driver_name}",
        "",
        f"**Verdict: {status}**",
        "",
        f"- Baseline run: `{report.baseline_run_id}`",
        f"- Replay run: `{report.replay_run_id}`",
        f"- Required checks: {sum(item.passed for item in required)}/{len(required)} passed",
        f"- Numeric metrics: {report.numeric['passed_count']}/{report.numeric['total_count']} within tolerance",
        f"- Top-neuron metrics: {report.ranking['neuron_metrics_passed']}/{report.ranking['neuron_metrics_total']} within tolerance",
        f"- Maximum absolute difference: {report.numeric['maximum_absolute_difference']:.8g}",
        f"- Maximum neuron-importance difference: {report.ranking['maximum_neuron_absolute_difference']:.8g}",
        f"- Top-{report.ranking['top_n']} neuron overlap: {report.ranking['overlap_fraction']:.3f}",
        f"- Shared-neuron sign agreement: {report.ranking['sign_agreement']:.3f}",
        f"- Mean rank displacement: {report.ranking['mean_rank_displacement']:.3f}",
        f"- Stable artifact hashes matched: {report.artifact_hashes['all_matched']}",
        "",
        "## Checks",
        "",
        "| Check | Required | Result |",
        "|---|---:|---:|",
    ]
    for item in report.checks:
        lines.append(
            f"| {item.name} | {'yes' if item.required else 'no'} | "
            f"{'pass' if item.passed else 'fail'} |"
        )
    lines.extend(
        [
            "",
            "## Replay boundary",
            "",
            "The comparison pins the experiment request, model snapshot, adapter, device, "
            "dtype, seed, algorithm, and selected package versions. Numeric observables and "
            "neuron rankings use the tolerances declared by the driver. Run IDs, timestamps, "
            "wall-clock durations, job/event records, and report-generation time are not "
            "treated as scientific equality conditions.",
            "",
        ]
    )
    if report.warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report.warnings)
        lines.append("")
    return "\n".join(lines)


def write_replay_report(bundle: ReplayBundle, report: ReplayReport) -> ReplayReport:
    bundle.report_directory.mkdir(parents=True, exist_ok=True)
    stem = f"replay-{report.replay_run_id}"
    json_path = bundle.report_directory / f"{stem}.json"
    markdown_path = bundle.report_directory / f"{stem}.md"
    relative_files = (
        str(json_path.relative_to(bundle.root)),
        str(markdown_path.relative_to(bundle.root)),
    )
    final_report = report.model_copy(update={"report_files": relative_files})
    _write_json_atomic(json_path, final_report)
    markdown_path.write_text(_markdown_report(final_report), encoding="utf-8")
    return final_report


def compact_replay_report(report: ReplayReport) -> ReplayOutcome:
    required = [item for item in report.checks if item.required]
    return ReplayOutcome(
        driver_name=report.driver_name,
        baseline_run_id=report.baseline_run_id,
        replay_run_id=report.replay_run_id,
        verdict=report.verdict,
        required_checks_passed=sum(item.passed for item in required),
        required_checks_total=len(required),
        numeric_metrics_passed=report.numeric["passed_count"],
        numeric_metrics_total=report.numeric["total_count"],
        neuron_metrics_passed=report.ranking["neuron_metrics_passed"],
        neuron_metrics_total=report.ranking["neuron_metrics_total"],
        maximum_absolute_difference=max(
            report.numeric["maximum_absolute_difference"],
            report.ranking["maximum_neuron_absolute_difference"],
        ),
        top_n=report.ranking["top_n"],
        top_n_overlap=report.ranking["overlap_fraction"],
        sign_agreement=report.ranking["sign_agreement"],
        mean_rank_displacement=report.ranking["mean_rank_displacement"],
        stable_artifact_hashes_matched=report.artifact_hashes["all_matched"],
        report_files=report.report_files,
        warnings=report.warnings,
    )


def compact_replay_identity(bundle: ReplayBundle) -> ReplayIdentity:
    """Stable, agent-readable identity used by docs and future orchestration."""
    return ReplayIdentity(
        driver_name=bundle.driver.name,
        driver=str(bundle.driver_path),
        spec=bundle.driver.spec,
        baseline=bundle.driver.baseline,
        baseline_exists=bundle.baseline_path.is_file(),
        science_hash=science_hash(bundle.spec),
        request_hash=request_hash(bundle.spec),
        reproducibility=bundle.driver.reproducibility,
        comparison=bundle.driver.comparison,
    )
