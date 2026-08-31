from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

from .contracts import (
    AttentionHeadInterventionRunSummary,
    AttentionHeadRankRunSummary,
    AttentionTraceRunSummary,
    AggregateLayerSummary,
    AggregateNeuronScore,
    NeuronCandidate,
    QueryEnvelope,
    DirectionInjectionRunSummary,
    FFNCouplingRunSummary,
    InterventionRunSummary,
    QualificationRunSummary,
    RankRunSummary,
    TrajectoryRunSummary,
    ReportReceipt,
    ResearchReport,
    RunOverview,
    RunManifest,
)


def _effect(importance: float) -> str:
    if importance > 0:
        return "toward_target"
    if importance < 0:
        return "toward_control"
    return "neutral"


def build_overview(
    *,
    run_id: str,
    summary: RankRunSummary | dict[str, Any],
    top_layers: int = 5,
    top_neurons: int = 10,
) -> RunOverview:
    parsed = (
        summary
        if isinstance(summary, RankRunSummary)
        else RankRunSummary.model_validate(summary)
    )
    layers = tuple(
        sorted(parsed.layers, key=lambda item: item.rms_mass, reverse=True)[:top_layers]
    )
    neurons = tuple(
        NeuronCandidate(
            **item.model_dump(mode="python"),
            observable_effect=_effect(item.importance_mean),
        )
        for item in parsed.neurons[:top_neurons]
    )
    return RunOverview(
        run_id=run_id,
        science_hash=parsed.science_hash,
        evidence_stage=parsed.evidence_stage,
        model=parsed.model,
        observable=parsed.observable,
        pair_count=parsed.pair_count,
        logical_forward_passes=parsed.logical_forward_passes,
        pairs=parsed.pairs,
        measured_delta_mean=parsed.measured_delta_mean,
        predicted_delta_mean=parsed.predicted_delta_mean,
        ffn_skip_mean=parsed.ffn_skip_mean,
        total_neuron_count=parsed.total_neuron_count,
        top_layers=layers,
        top_neurons=neurons,
        qualification=parsed.qualification,
        claims=parsed.claims,
        warnings=parsed.warnings,
    )


def query_envelope(
    *,
    run_id: str,
    query: str,
    items: list[dict[str, Any]],
    source_count: int,
    matched_count: int | None = None,
    parameters: dict[str, Any] | None = None,
    sort: str | None = None,
) -> QueryEnvelope:
    return QueryEnvelope(
        run_id=run_id,
        query=query,
        parameters=parameters or {},
        sort=sort,
        source_count=source_count,
        matched_count=matched_count if matched_count is not None else len(items),
        returned_count=len(items),
        items=tuple(items),
    )


def _format_effect(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:+.6g}"


def build_research_report(
    *,
    run_id: str,
    manifest: RunManifest,
    summary: dict[str, Any],
) -> ResearchReport:
    """Build a conservative, schema-driven narrative from immutable evidence."""

    key_results: list[str] = []
    next_steps: list[str] = []
    if manifest.run_kind == "rank":
        parsed = RankRunSummary.model_validate(summary)
        qualification = parsed.qualification
        qualified = (
            f"{qualification.informative_pairs}/{parsed.pair_count} informative pairs"
            if qualification is not None
            else "qualification unavailable"
        )
        key_results.extend(
            (
                f"Observable movement mean: {_format_effect(parsed.measured_delta_mean)}.",
                f"FFN additive prediction mean: {_format_effect(parsed.predicted_delta_mean)}; FFN/Skip mean: {_format_effect(parsed.ffn_skip_mean)}.",
                f"First-token qualification: {qualified}.",
            )
        )
        if parsed.layers:
            layer = max(parsed.layers, key=lambda item: item.rms_mass)
            key_results.append(
                f"Highest aggregate FFN mass is layer {layer.layer} (RMS mass {layer.rms_mass:.6g})."
            )
        if parsed.neurons:
            neuron = parsed.neurons[0]
            key_results.append(
                f"Leading ranked unit is L{neuron.layer}/N{neuron.neuron} (importance RMS {neuron.importance_rms:.6g}, sign consistency {neuron.sign_consistency:.3f})."
            )
        headline = (
            "Replicated observational neuron ranking"
            if parsed.pair_count > 1
            else "Exploratory observational neuron ranking"
        )
        next_steps.extend(
            (
                "Run generated-behavior qualification with a predeclared evaluator.",
                "Test ranked neurons with dose sweeps and matched random controls.",
            )
        )
        claims = parsed.claims
        limitations = parsed.warnings
    elif manifest.run_kind == "trajectory":
        parsed = TrajectoryRunSummary.model_validate(summary)
        key_results.append(
            f"Decoded {parsed.pair_count} paired trajectories with {parsed.logical_forward_passes} logical forward passes."
        )
        leading = [
            (pair, pair.transitions[0])
            for pair in parsed.pairs
            if pair.transitions
        ]
        if leading:
            pair, transition = max(
                leading, key=lambda item: item[1].absolute_change
            )
            key_results.append(
                f"Largest suggested transition is {pair.pair_id} at L{transition.layer}/{transition.checkpoint} "
                f"(paired gap change {transition.pair_delta_change:+.6g})."
            )
        headline = "Observational native paired trajectory"
        next_steps.extend(
            (
                "Confirm a transition band before scoping component analysis.",
                "Test candidate states and components with matched causal interventions.",
            )
        )
        claims = parsed.claims
        limitations = parsed.warnings
    elif manifest.run_kind == "ffn_coupling":
        parsed = FFNCouplingRunSummary.model_validate(summary)
        key_results.append(
            f"Compared direct, native-local, and downstream-gradient coupling for {parsed.total_neuron_count} neurons."
        )
        if parsed.neurons:
            neuron = parsed.neurons[0]
            key_results.append(
                f"Leading downstream-sensitive unit is L{neuron.layer}/N{neuron.neuron} "
                f"(gradient importance RMS {neuron.downstream_importance_rms:.6g}, "
                f"direct/gradient sign agreement {neuron.direct_downstream_sign_agreement:.3f})."
            )
        headline = "Observational layer-aware FFN coupling"
        next_steps.extend(
            (
                "Compare top gradient-ranked neurons with same-layer random controls.",
                "Treat direct-versus-gradient disagreement as a hypothesis about downstream transformation.",
            )
        )
        claims = parsed.claims
        limitations = parsed.warnings
    elif manifest.run_kind == "qualify":
        parsed = QualificationRunSummary.model_validate(summary)
        aggregate = parsed.aggregate
        key_results.extend(
            (
                f"Generated-behavior qualification found {aggregate.informative_pairs} informative, {aggregate.weak_pairs} weak, and {aggregate.invalid_pairs} invalid pairs.",
                f"Claim eligibility is {str(aggregate.claim_eligible).lower()} under evaluator kind {parsed.evaluator.kind}.",
            )
        )
        headline = (
            "Observable qualified against generated behavior"
            if aggregate.claim_eligible
            else "Observable qualification did not clear the claim gate"
        )
        next_steps.append(
            "Run interventions only on informative pairs and retain this run as qualification lineage."
        )
        claims = parsed.claims
        limitations = parsed.warnings
    elif manifest.run_kind == "intervention":
        parsed = InterventionRunSummary.model_validate(summary)
        controlled = [
            item
            for item in parsed.doses
            if item.controlled_absolute_effect is not None
        ]
        strongest = max(
            controlled,
            key=lambda item: item.controlled_absolute_effect or float("-inf"),
            default=None,
        )
        key_results.append(
            f"Tested {len(parsed.selected_neurons)} ranked neurons with {parsed.operation.mode} over {len(parsed.doses)} split/condition/dose summaries."
        )
        if strongest is not None:
            key_results.append(
                f"Largest selected-minus-random absolute effect was {_format_effect(strongest.controlled_absolute_effect)} at N={strongest.neuron_count}, strength={strongest.strength:g}, split={strongest.split}, condition={strongest.condition}."
            )
        if parsed.causal_width:
            widths = ", ".join(
                f"{item.split}/{item.condition}/strength={item.strength:g}:N{item.width_at_90_percent}"
                for item in parsed.causal_width
            )
            key_results.append(f"Estimated 90% causal widths: {widths}.")
        headline = "Controlled FFN intervention evidence"
        next_steps.extend(
            (
                "Replicate the controlled effect on held-out perturbations.",
                "Inspect collateral observables and additivity before assigning a narrow circuit claim.",
            )
        )
        claims = parsed.claims
        limitations = parsed.warnings
    elif manifest.run_kind == "direction":
        parsed = DirectionInjectionRunSummary.model_validate(summary)
        controlled = [
            item
            for item in parsed.doses
            if item.controlled_absolute_effect is not None
        ]
        strongest = max(
            controlled,
            key=lambda item: item.controlled_absolute_effect or float("-inf"),
            default=None,
        )
        key_results.append(
            f"Swept {len(parsed.layers)} layers and {len(parsed.betas)} beta values with direction norm {parsed.behavioral_direction_norm:.6g}."
        )
        if strongest is not None:
            key_results.append(
                f"Largest direction-minus-random absolute effect was {_format_effect(strongest.controlled_absolute_effect)} at layer {strongest.layer}, beta={strongest.beta:g}, split={strongest.split}, condition={strongest.condition}."
            )
        headline = "Residual-direction controllability evidence"
        next_steps.extend(
            (
                "Compare the layer sweep with FFN intervention effects.",
                "Do not infer neuron localization from direction controllability alone.",
            )
        )
        claims = parsed.claims
        limitations = parsed.warnings
    elif manifest.run_kind == "attention_rank":
        parsed = AttentionHeadRankRunSummary.model_validate(summary)
        key_results.append(
            f"Ranked {parsed.total_head_count} output heads across {parsed.pair_count} prompt pairs."
        )
        if parsed.heads:
            head = parsed.heads[0]
            key_results.append(
                f"Leading direct-logit head is L{head.layer}/H{head.head} "
                f"(effect RMS {head.direct_effect_rms:.6g}, sign consistency {head.sign_consistency:.3f})."
            )
        if parsed.layers:
            layer = max(parsed.layers, key=lambda item: item.rms_mass)
            key_results.append(
                f"Highest attention-head RMS mass is layer {layer.layer} ({layer.rms_mass:.6g})."
            )
        headline = "Observational attention-head routing hypotheses"
        next_steps.extend(
            (
                "Run head-output dose sweeps with same-layer random controls.",
                "Trace token routes only for intervention-supported heads.",
            )
        )
        claims = parsed.claims
        limitations = parsed.warnings
    elif manifest.run_kind == "attention_intervention":
        parsed = AttentionHeadInterventionRunSummary.model_validate(summary)
        controlled = [
            item
            for item in parsed.doses
            if item.controlled_absolute_effect is not None
        ]
        strongest = max(
            controlled,
            key=lambda item: item.controlled_absolute_effect or float("-inf"),
            default=None,
        )
        key_results.append(
            f"Tested {len(parsed.selected_heads)} attention heads over {len(parsed.doses)} dose summaries."
        )
        if strongest is not None:
            key_results.append(
                "Largest selected-minus-random absolute effect was "
                f"{_format_effect(strongest.controlled_absolute_effect)} at "
                f"H={strongest.head_count}, strength={strongest.strength:g}, "
                f"split={strongest.split}, condition={strongest.condition}."
            )
        headline = "Controlled attention-head intervention evidence"
        next_steps.extend(
            (
                "Inspect token contributions into supported heads with eager-attention reconstruction checks.",
                "Test sender-to-receiver paths under exact token alignment.",
            )
        )
        claims = parsed.claims
        limitations = parsed.warnings
    else:
        parsed = AttentionTraceRunSummary.model_validate(summary)
        if parsed.trace_kind == "token_edges":
            key_results.append(
                f"Retained {len(parsed.token_edges)} highest-magnitude token-to-head edges."
            )
            if parsed.token_edges:
                edge = parsed.token_edges[0]
                key_results.append(
                    f"Leading edge is {edge.pair_id}/{edge.condition} token "
                    f"{edge.source_position} into L{edge.layer}/H{edge.head} "
                    f"(direct effect {edge.direct_effect:+.6g})."
                )
            headline = "Observational attention token-route hypotheses"
            next_steps.append(
                "Use exact-alignment two-stage patching to test selected sender-to-receiver routes."
            )
        else:
            selected = [
                item for item in parsed.paths if item.arm == "selected_path"
            ]
            strongest = max(
                selected,
                key=lambda item: abs(item.path_specific_effect),
                default=None,
            )
            key_results.append(
                f"Tested {len(selected)} selected sender-to-receiver path observations."
            )
            if strongest is not None:
                key_results.append(
                    f"Largest selected path effect is L{strongest.sender.layer}/H{strongest.sender.head} "
                    f"to L{strongest.receiver.layer}/H{strongest.receiver.head}: "
                    f"{strongest.path_specific_effect:+.6g}."
                )
            headline = (
                "Supported local attention path-patching evidence"
                if any(claim.status == "supported" for claim in parsed.claims)
                else "Exploratory attention path-patching result"
            )
            next_steps.append(
                "Replicate supported paths on held-out prompt pairs and alternate observables."
            )
        claims = parsed.claims
        limitations = parsed.warnings

    return ResearchReport(
        run_id=run_id,
        run_kind=manifest.run_kind,
        evidence_stage=manifest.evidence_stage,
        parent_run_ids=manifest.parent_run_ids,
        headline=headline,
        key_results=tuple(key_results),
        claims=claims,
        limitations=tuple(dict.fromkeys(limitations)),
        recommended_next_steps=tuple(next_steps),
    )


def render_research_report(report: ResearchReport) -> str:
    lines = [
        f"# {report.headline}",
        "",
        f"- Run: `{report.run_id}`",
        f"- Kind: `{report.run_kind}`",
        f"- Evidence stage: `{report.evidence_stage}`",
    ]
    if report.parent_run_ids:
        lines.append(f"- Parent runs: {', '.join(f'`{item}`' for item in report.parent_run_ids)}")
    lines.extend(("", "## Key results", ""))
    lines.extend(f"- {item}" for item in report.key_results)
    lines.extend(("", "## Claims", ""))
    if report.claims:
        for claim in report.claims:
            lines.append(
                f"- **{claim.status} / {claim.claim_type}:** {claim.statement}"
            )
            lines.extend(f"  - Limitation: {item}" for item in claim.limitations)
    else:
        lines.append("- No formal claim was emitted.")
    lines.extend(("", "## Limitations", ""))
    lines.extend(f"- {item}" for item in report.limitations)
    if not report.limitations:
        lines.append("- No additional run warning was recorded; method-level limits still apply.")
    lines.extend(("", "## Recommended next steps", ""))
    lines.extend(f"- {item}" for item in report.recommended_next_steps)
    return "\n".join(lines) + "\n"


def write_research_report(
    *,
    report: ResearchReport,
    output_directory: Path,
) -> ReportReceipt:
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "report.json"
    markdown_path = output_directory / "report.md"
    json_text = json.dumps(
        report.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    markdown_text = render_research_report(report)
    for path, content in ((json_path, json_text), (markdown_path, markdown_text)):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    return ReportReceipt(
        run_id=report.run_id,
        json_path=str(json_path.resolve()),
        markdown_path=str(markdown_path.resolve()),
        json_sha256=sha256(json_text.encode("utf-8")).hexdigest(),
        markdown_sha256=sha256(markdown_text.encode("utf-8")).hexdigest(),
    )
