from __future__ import annotations

from dataclasses import dataclass
from html import escape
import math
from pathlib import Path
from typing import Iterable, Sequence

from .contracts import InterventionRunSummary, TrajectoryRunSummary


CHECKPOINT_ORDER = {"block_input": 0, "post_attention": 1, "post_ffn": 2}
CHECKPOINT_LABEL = {
    "block_input": "input",
    "post_attention": "attention",
    "post_ffn": "FFN",
}
METHOD_LABEL = {
    "direct_structural": "Direct readout",
    "downstream_endpoint_gradient": "Downstream gradient",
    "direct_downstream_overlap": "Direct/downstream overlap",
}
METHOD_COLOR = {
    "direct_structural": "#2563eb",
    "downstream_endpoint_gradient": "#0f9f8f",
    "direct_downstream_overlap": "#d97706",
}


@dataclass(frozen=True)
class ChartSeries:
    name: str
    values: tuple[float, ...]
    color: str
    dashed: bool = False
    width: float = 2.4


def _finite(values: Iterable[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError("trajectory visualization requires finite, non-empty series")
    return result


def _line_chart(
    *,
    chart_id: str,
    title: str,
    description: str,
    labels: Sequence[str],
    layers: Sequence[int],
    checkpoints: Sequence[str],
    series: Sequence[ChartSeries],
    marker_index: int | None = None,
    marker_label: str | None = None,
) -> str:
    if not labels or any(len(item.values) != len(labels) for item in series):
        raise ValueError("chart labels and series must have matching non-zero lengths")
    width, height = 1160, 390
    left, right, top, bottom = 74, 24, 34, 62
    plot_width = width - left - right
    plot_height = height - top - bottom
    values = [value for item in series for value in item.values]
    y_min = min(values + [0.0])
    y_max = max(values + [0.0])
    span = max(y_max - y_min, 1e-6)
    y_min -= span * 0.08
    y_max += span * 0.08

    def x_at(index: int) -> float:
        return left + (plot_width * index / max(len(labels) - 1, 1))

    def y_at(value: float) -> float:
        return top + plot_height * (y_max - value) / (y_max - y_min)

    grid = []
    for step in range(6):
        value = y_min + (y_max - y_min) * step / 5
        y = y_at(value)
        grid.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" class="grid" />'
            f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" class="axis-label">{value:.2f}</text>'
        )
    tick_indexes = [
        index
        for index, (layer, checkpoint) in enumerate(zip(layers, checkpoints))
        if checkpoint == "post_ffn" and (layer % 4 == 0 or index == len(labels) - 1)
    ]
    ticks = [
        f'<line x1="{x_at(index):.2f}" y1="{height-bottom}" x2="{x_at(index):.2f}" y2="{height-bottom+5}" class="axis" />'
        f'<text x="{x_at(index):.2f}" y="{height-bottom+22}" text-anchor="middle" class="axis-label">L{layers[index]}</text>'
        for index in tick_indexes
    ]
    paths = []
    legend = []
    for position, item in enumerate(series):
        points = " ".join(
            f"{x_at(index):.2f},{y_at(value):.2f}"
            for index, value in enumerate(item.values)
        )
        dash = ' stroke-dasharray="7 6"' if item.dashed else ""
        paths.append(
            f'<polyline points="{points}" fill="none" stroke="{item.color}" '
            f'stroke-width="{item.width}" stroke-linejoin="round" stroke-linecap="round"{dash} />'
        )
        legend_x = left + (position % 3) * 315
        legend_y = height - 10 + (position // 3) * 19
        legend.append(
            f'<line x1="{legend_x}" y1="{legend_y-4}" x2="{legend_x+26}" y2="{legend_y-4}" '
            f'stroke="{item.color}" stroke-width="{item.width}"{dash} />'
            f'<text x="{legend_x+34}" y="{legend_y}" class="legend-label">{escape(item.name)}</text>'
        )
    marker = ""
    if marker_index is not None and 0 <= marker_index < len(labels):
        x = x_at(marker_index)
        marker = (
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}" class="marker" />'
            f'<text x="{min(x+8, width-230):.2f}" y="{top+15}" class="marker-label">{escape(marker_label or labels[marker_index])}</text>'
        )
    return f"""
      <section class="panel chart-panel" aria-labelledby="{chart_id}-heading">
        <div class="panel-heading"><div><p class="eyebrow">TRAJECTORY</p><h2 id="{chart_id}-heading">{escape(title)}</h2></div></div>
        <p class="caption">{escape(description)}</p>
        <svg role="img" aria-labelledby="{chart_id}-title {chart_id}-desc" viewBox="0 0 {width} {height + max(0, ((len(series)-1)//3)*19)}">
          <title id="{chart_id}-title">{escape(title)}</title>
          <desc id="{chart_id}-desc">{escape(description)}</desc>
          {''.join(grid)}
          <line x1="{left}" y1="{y_at(0):.2f}" x2="{width-right}" y2="{y_at(0):.2f}" class="zero" />
          <line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis" />
          <line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis" />
          {''.join(ticks)}
          {''.join(paths)}
          {marker}
          {''.join(legend)}
        </svg>
      </section>
    """


def _selected_overlay(summary: InterventionRunSummary, pair_id: str):
    rows = [
        item
        for item in summary.trajectory_overlays
        if item.pair_id == pair_id and item.arm == "selected"
    ]
    if not rows:
        raise ValueError(
            f"intervention has no selected trajectory overlay for pair {pair_id!r}"
        )
    width = max(item.neuron_count for item in rows)
    at_width = [item for item in rows if item.neuron_count == width]
    strength = max(
        {item.strength for item in at_width}, key=lambda value: (abs(value), value)
    )
    selected = sorted(
        (item for item in at_width if item.strength == strength),
        key=lambda item: (item.layer, CHECKPOINT_ORDER[item.checkpoint]),
    )
    controls_by_key: dict[tuple[int, str], list[float]] = {}
    control_samples: set[int] = set()
    for item in summary.trajectory_overlays:
        if (
            item.pair_id != pair_id
            or item.arm != "matched_random"
            or item.neuron_count != width
            or item.strength != strength
        ):
            continue
        controls_by_key.setdefault((item.layer, item.checkpoint), []).append(
            item.gap_effect
        )
        if item.control_sample is not None:
            control_samples.add(item.control_sample)
    control_mean = tuple(
        sum(values) / len(values)
        for item in selected
        if (values := controls_by_key.get((item.layer, item.checkpoint)))
    )
    if len(control_mean) != len(selected):
        raise ValueError("matched-random trajectory overlay is incomplete")
    return selected, control_mean, width, strength, len(control_samples)


def _claim_status(summary: InterventionRunSummary) -> str:
    relevant = [
        item.status
        for item in summary.claims
        if item.claim_type in {"necessity", "sufficiency"}
    ]
    return relevant[0] if relevant else "not claimed"


def render_trajectory_visualization(
    *,
    trajectory_run_id: str,
    trajectory: TrajectoryRunSummary,
    intervention_runs: Sequence[tuple[str, InterventionRunSummary]],
    pair_id: str | None = None,
) -> str:
    pair = next(
        (item for item in trajectory.pairs if item.pair_id == pair_id),
        trajectory.pairs[0] if pair_id is None and trajectory.pairs else None,
    )
    if pair is None:
        raise ValueError(f"trajectory has no pair {pair_id!r}")
    pair_id = pair.pair_id
    for run_id, intervention in intervention_runs:
        if intervention.trajectory_run_id != trajectory_run_id:
            raise ValueError(
                f"intervention {run_id!r} does not descend from trajectory {trajectory_run_id!r}"
            )

    checkpoints = list(pair.checkpoints)
    labels = [f"L{item.layer} {CHECKPOINT_LABEL[item.checkpoint]}" for item in checkpoints]
    layers = [item.layer for item in checkpoints]
    checkpoint_names = [item.checkpoint for item in checkpoints]
    strongest = pair.transitions[0] if pair.transitions else None
    marker_index = next(
        (
            index
            for index, item in enumerate(checkpoints)
            if strongest
            and item.layer == strongest.layer
            and item.checkpoint == strongest.checkpoint
        ),
        None,
    )
    baseline_chart = _line_chart(
        chart_id="paired-readout",
        title="Where the paired prediction separates",
        description=(
            "Native final-norm and LM-head decoding at every block input, post-attention, and post-FFN checkpoint. "
            "These curves locate decodable change; they are observational, not causal attribution."
        ),
        labels=labels,
        layers=layers,
        checkpoints=checkpoint_names,
        series=(
            ChartSeries("Original target-control gap", _finite(item.original_gap for item in checkpoints), "#2563eb"),
            ChartSeries("Perturbed target-control gap", _finite(item.perturbed_gap for item in checkpoints), "#c24175"),
            ChartSeries("Paired delta", _finite(item.pair_delta for item in checkpoints), "#0f9f8f", dashed=True),
        ),
        marker_index=marker_index,
        marker_label=(
            f"largest change: L{strongest.layer} {CHECKPOINT_LABEL[strongest.checkpoint]} "
            f"({strongest.pair_delta_change:+.2f})"
            if strongest
            else None
        ),
    )

    overlays = []
    method_cards = []
    dose_rows = []
    overlay_labels: list[str] | None = None
    overlay_layers: list[int] | None = None
    overlay_checkpoints: list[str] | None = None
    for run_id, intervention in intervention_runs:
        selected, controls, width, strength, control_count = _selected_overlay(
            intervention, pair_id
        )
        method = intervention.candidate_score_method
        method_label = METHOD_LABEL[method]
        color = METHOD_COLOR[method]
        current_labels = [
            f"L{item.layer} {CHECKPOINT_LABEL[item.checkpoint]}" for item in selected
        ]
        if overlay_labels is None:
            overlay_labels = current_labels
            overlay_layers = [item.layer for item in selected]
            overlay_checkpoints = [item.checkpoint for item in selected]
        elif current_labels != overlay_labels:
            raise ValueError("intervention trajectories do not share checkpoint alignment")
        overlays.extend(
            (
                ChartSeries(method_label, _finite(item.gap_effect for item in selected), color, width=3.0),
                ChartSeries(f"{method_label} random mean", _finite(controls), color, dashed=True, width=1.6),
            )
        )
        claim = _claim_status(intervention)
        objective = intervention.candidate_ranking_objective
        method_cards.append(
            f"<article><span class=\"method-dot\" style=\"--method:{color}\"></span>"
            f"<div><strong>{escape(method_label)}</strong><small>{escape(objective)} · width {width} · strength {strength:g}</small></div>"
            f"<b class=\"claim {escape(claim)}\">{escape(claim)}</b></article>"
        )
        for dose in intervention.doses:
            dose_rows.append(
                "<tr>"
                f"<td>{escape(method_label)}</td><td>{escape(dose.split)}</td>"
                f"<td>{dose.neuron_count}</td><td>{dose.selected_absolute_effect_mean:.4f}</td>"
                f"<td>{(dose.random_absolute_effect_mean or 0.0):.4f}</td>"
                f"<td><strong>{(dose.controlled_absolute_effect or 0.0):+.4f}</strong></td>"
                f"<td>{dose.random_observation_count}</td></tr>"
            )
        method_cards[-1] = method_cards[-1].replace(
            "</small>", f" · {control_count} control draws</small>"
        )

    causal_chart = ""
    if overlays and overlay_labels and overlay_layers and overlay_checkpoints:
        causal_chart = _line_chart(
            chart_id="causal-propagation",
            title="Where controlled patch effects become decodable",
            description=(
                "Solid lines are the widest predeclared selected-neuron patch; dashed lines are same-layer matched-random means. "
                "The effect at a checkpoint is intervention gap minus its unmodified baseline. It can grow, shrink, or reverse downstream and is not conserved flow."
            ),
            labels=overlay_labels,
            layers=overlay_layers,
            checkpoints=overlay_checkpoints,
            series=tuple(overlays),
        )

    transitions = "".join(
        f"<li><span>#{item.rank}</span><strong>L{item.layer} {escape(CHECKPOINT_LABEL[item.checkpoint])}</strong>"
        f"<b>{item.pair_delta_change:+.4f}</b></li>"
        for item in pair.transitions[:6]
    )
    model_id = str(trajectory.model.get("model_id") or trajectory.model.get("id") or "unknown")
    revision = str(
        trajectory.model.get("resolved_revision")
        or trajectory.model.get("revision")
        or "unknown"
    )
    observable = str(trajectory.observable.get("name") or "target-control gap")
    intervention_ids = ", ".join(run_id for run_id, _summary in intervention_runs)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Paired trajectory evidence · {escape(pair_id)}</title>
  <style>
    :root {{ color-scheme: light; --ink:#17252b; --muted:#61747b; --line:#d9e2e4; --paper:#fff; --wash:#f3f7f6; --teal:#0f9f8f; --pink:#c24175; }}
    * {{ box-sizing:border-box; }} body {{ margin:0; background:#edf3f2; color:var(--ink); font:15px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    main {{ max-width:1240px; margin:0 auto; padding:42px 28px 64px; }}
    header {{ padding:34px; border:1px solid var(--line); background:linear-gradient(135deg,#fff 62%,#e4f3f0); box-shadow:0 14px 40px rgba(25,53,60,.08); }}
    h1 {{ max-width:840px; margin:8px 0 12px; font-size:clamp(32px,5vw,62px); line-height:1.02; letter-spacing:-.045em; }} h2 {{ margin:0; font-size:22px; letter-spacing:-.02em; }}
    .lede {{ max-width:820px; color:var(--muted); font-size:17px; }} .eyebrow {{ margin:0; color:var(--teal); font:700 11px/1.2 ui-monospace,SFMono-Regular,Menlo,monospace; letter-spacing:.14em; }}
    .meta {{ display:grid; grid-template-columns:repeat(4,1fr); gap:1px; margin-top:28px; background:var(--line); border:1px solid var(--line); }} .meta div {{ background:#fff; padding:15px; min-width:0; }}
    dt {{ color:var(--muted); font:700 10px/1.2 ui-monospace,SFMono-Regular,Menlo,monospace; letter-spacing:.1em; text-transform:uppercase; }} dd {{ margin:5px 0 0; font-weight:650; overflow-wrap:anywhere; }}
    .panel {{ margin-top:18px; padding:24px; border:1px solid var(--line); background:var(--paper); box-shadow:0 8px 24px rgba(25,53,60,.05); }} .caption {{ margin:7px 0 10px; color:var(--muted); max-width:960px; }}
    svg {{ width:100%; height:auto; overflow:visible; }} .grid {{ stroke:#e6ecee; stroke-width:1; }} .axis {{ stroke:#9aaeb3; stroke-width:1; }} .zero {{ stroke:#81969b; stroke-width:1.2; }} .axis-label,.legend-label {{ fill:#61747b; font:11px ui-monospace,SFMono-Regular,Menlo,monospace; }} .marker {{ stroke:#9b6b12; stroke-width:1.2; stroke-dasharray:3 4; }} .marker-label {{ fill:#7a5310; font:700 11px ui-monospace,SFMono-Regular,Menlo,monospace; }}
    .evidence-grid {{ display:grid; grid-template-columns:.8fr 1.2fr; gap:18px; }} .transitions ol,.methods {{ list-style:none; padding:0; margin:15px 0 0; }} .transitions li,.methods article {{ display:grid; align-items:center; gap:12px; padding:10px 0; border-top:1px solid #edf1f2; }} .transitions li {{ grid-template-columns:38px 1fr auto; }} .transitions span {{ color:var(--muted); font:11px ui-monospace,SFMono-Regular,Menlo,monospace; }} .transitions b {{ color:var(--teal); font:12px ui-monospace,SFMono-Regular,Menlo,monospace; }}
    .methods article {{ grid-template-columns:12px 1fr auto; }} .methods small {{ display:block; color:var(--muted); }} .method-dot {{ width:9px; height:9px; border-radius:50%; background:var(--method); }} .claim {{ padding:4px 8px; border-radius:999px; background:#edf1f2; color:#4c6167; font-size:11px; }} .claim.supported {{ background:#dff4ed; color:#087363; }}
    .table-wrap {{ overflow:auto; }} table {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }} th,td {{ padding:10px 12px; text-align:right; border-bottom:1px solid #e8edef; white-space:nowrap; }} th:first-child,td:first-child {{ text-align:left; }} th {{ color:var(--muted); font:700 10px ui-monospace,SFMono-Regular,Menlo,monospace; letter-spacing:.08em; text-transform:uppercase; }}
    .interpretation {{ border-left:4px solid var(--teal); }} code {{ font:12px ui-monospace,SFMono-Regular,Menlo,monospace; color:#34515a; overflow-wrap:anywhere; }} footer {{ margin-top:24px; color:var(--muted); font-size:12px; }}
    @media (max-width:800px) {{ main {{ padding:18px 12px 40px; }} header,.panel {{ padding:18px; }} .meta {{ grid-template-columns:1fr 1fr; }} .evidence-grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body><main>
  <header>
    <p class="eyebrow">PERTURBATION PROBING · AUDITABLE EXAMPLE</p>
    <h1>From paired prediction change to controlled causal propagation.</h1>
    <p class="lede">This figure keeps two questions separate: where the original and perturbed prompts become distinguishable, and where a controlled neuron patch first changes the decoded observable.</p>
    <dl class="meta">
      <div><dt>Pair</dt><dd>{escape(pair_id)} · {escape(pair.split)}</dd></div>
      <div><dt>Observable</dt><dd>{escape(observable)}</dd></div>
      <div><dt>Model</dt><dd>{escape(model_id)}</dd></div>
      <div><dt>Revision</dt><dd>{escape(revision[:14])}</dd></div>
    </dl>
  </header>
  {baseline_chart}
  <div class="evidence-grid">
    <section class="panel transitions"><p class="eyebrow">OBSERVATIONAL SUGGESTIONS</p><h2>Largest paired transitions</h2><ol>{transitions}</ol></section>
    <section class="panel"><p class="eyebrow">CONTROLLED FOLLOW-UP</p><h2>Intervention lineage and claims</h2><div class="methods">{''.join(method_cards) or '<p>No intervention overlays supplied.</p>'}</div></section>
  </div>
  {causal_chart}
  <section class="panel"><p class="eyebrow">DOSE × SPLIT</p><h2>Selected effects versus matched controls</h2><p class="caption">Controlled effect is selected absolute effect minus matched-random mean absolute effect. Held-out here means evaluation on a predeclared pair, not population-level generalization.</p><div class="table-wrap"><table><thead><tr><th>Candidate method</th><th>Split</th><th>Width</th><th>Selected |effect|</th><th>Random |effect|</th><th>Controlled</th><th>Random obs.</th></tr></thead><tbody>{''.join(dose_rows)}</tbody></table></div></section>
  <section class="panel interpretation"><p class="eyebrow">INTERPRETATION BOUNDARY</p><h2>Read the sequence, not just the peak</h2><p>The paired readout is an observational localization aid. The patch overlays become local causal evidence only through each immutable intervention run's recorded qualification, matched controls, and claim status. A downstream rise does not mean that the intervened signal is conserved; later blocks can transform or amplify it.</p></section>
  <footer>Trajectory run <code>{escape(trajectory_run_id)}</code>{(' · intervention runs <code>' + escape(intervention_ids) + '</code>') if intervention_ids else ''}. Generated from verified immutable summaries by <code>probe runs trajectory-visualize</code>.</footer>
</main></body></html>
"""


def write_trajectory_visualization(path: Path, html: str) -> Path:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, encoding="utf-8")
    return destination


__all__ = ["render_trajectory_visualization", "write_trajectory_visualization"]
