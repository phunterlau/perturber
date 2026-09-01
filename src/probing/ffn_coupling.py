from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import fmean

import torch

from .contracts import (
    FFNCouplingLayerSummary,
    FFNCouplingNeuronScore,
    FFNCouplingPairSummary,
    FFNCouplingRunSummary,
    FFNCouplingSpec,
    RankRunSummary,
    RankSpec,
)
from .engine import ProbeEngine
from .observables import resolve_observable


@dataclass(frozen=True)
class FFNCouplingComputation:
    summary: FFNCouplingRunSummary
    tensors: dict[str, torch.Tensor]


def ffn_coupling_plan_counts(
    *, parent_spec: RankSpec, parent_summary: RankRunSummary, spec: FFNCouplingSpec
) -> tuple[int, int, int]:
    available_pairs = {pair.pair_id for pair in parent_summary.pairs}
    requested = spec.pair_ids or tuple(pair.pair_id for pair in parent_summary.pairs)
    unknown = sorted(set(requested) - available_pairs)
    if unknown:
        raise ValueError(f"FFN coupling references unknown parent pairs: {unknown}")
    split_by_id = {pair.pair_id: pair.split for pair in parent_summary.pairs}
    discovery = [pair_id for pair_id in requested if split_by_id[pair_id] == "discovery"]
    if not discovery:
        raise ValueError(
            "FFN coupling candidate ranking requires at least one discovery pair"
        )
    if spec.layers:
        invalid = [layer for layer in spec.layers if layer >= parent_summary.model["layer_count"]]
        if invalid:
            raise ValueError(f"FFN coupling layers are out of range: {invalid}")
    pair_count = len(requested)
    return pair_count, 2 * pair_count, 2 * pair_count


def _sign_consistency(values: torch.Tensor) -> torch.Tensor:
    mean_sign = values.mean(dim=0).sign()
    agreement = (values.sign() == mean_sign.unsqueeze(0)).float().mean(dim=0)
    return torch.where(mean_sign == 0, torch.zeros_like(agreement), agreement)


def run_ffn_coupling(
    *,
    engine: ProbeEngine,
    parent_spec: RankSpec,
    parent_summary: RankRunSummary,
    parent_tensors: dict[str, torch.Tensor],
    spec: FFNCouplingSpec,
    science_hash: str,
) -> FFNCouplingComputation:
    observable = resolve_observable(engine.adapter.tokenizer, parent_spec.observable)
    selected_ids = set(spec.pair_ids) if spec.pair_ids else None
    parent_index = {pair.pair_id: index for index, pair in enumerate(parent_summary.pairs)}
    pair_specs = {pair.id: pair for pair in parent_spec.pairs}
    selected_pairs = [
        pair
        for pair in parent_summary.pairs
        if selected_ids is None or pair.pair_id in selected_ids
    ]
    layer_count = int(parent_summary.model["layer_count"])
    layers = spec.layers or tuple(range(layer_count))

    downstream_couplings: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    downstream_importances: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    native_couplings: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    native_importances: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    direct_importances: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    activation_deltas: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    pair_results: list[FFNCouplingPairSummary] = []
    tensors: dict[str, torch.Tensor] = {}

    for pair_summary in selected_pairs:
        pair = pair_specs[pair_summary.pair_id]
        pair_number = parent_index[pair.id]
        original_ids, original_tokens = engine.adapter.prepare_prompt(
            text=pair.original,
            messages=tuple(item.model_dump(mode="json") for item in pair.original_messages),
            tools=pair.tools,
            chat_template=parent_spec.model.chat_template,
            enable_thinking=parent_spec.model.enable_thinking,
        )
        perturbed_ids, perturbed_tokens = engine.adapter.prepare_prompt(
            text=pair.perturbed,
            messages=tuple(item.model_dump(mode="json") for item in pair.perturbed_messages),
            tools=pair.tools,
            chat_template=parent_spec.model.chat_template,
            enable_thinking=parent_spec.model.enable_thinking,
        )
        original = engine.adapter.forward_residual_gradients(
            original_ids, original_tokens, spec.position, observable
        )
        perturbed = engine.adapter.forward_residual_gradients(
            perturbed_ids, perturbed_tokens, spec.position, observable
        )
        original_downstream = engine.adapter.layer_couplings(original.gradients)
        perturbed_downstream = engine.adapter.layer_couplings(perturbed.gradients)
        original_native_gradients = tuple(
            engine.adapter.native_residual_gradient(residual, observable)
            for residual in original.residuals
        )
        perturbed_native_gradients = tuple(
            engine.adapter.native_residual_gradient(residual, observable)
            for residual in perturbed.residuals
        )
        original_native = engine.adapter.layer_couplings(original_native_gradients)
        perturbed_native = engine.adapter.layer_couplings(perturbed_native_gradients)

        pair_results.append(
            FFNCouplingPairSummary(
                pair_id=pair.id,
                split=pair.split,
                original_gradient_norm_mean=fmean(
                    float(original.gradients[layer].norm().item()) for layer in layers
                ),
                perturbed_gradient_norm_mean=fmean(
                    float(perturbed.gradients[layer].norm().item()) for layer in layers
                ),
            )
        )
        for layer in layers:
            delta = parent_tensors[
                f"activation_delta.pair_{pair_number}.layer_{layer}"
            ].detach().float().cpu()
            direct = parent_tensors[f"coupling.layer_{layer}"].detach().float().cpu()
            downstream = 0.5 * (
                original_downstream[layer] + perturbed_downstream[layer]
            )
            native = 0.5 * (original_native[layer] + perturbed_native[layer])
            if pair.split == "discovery":
                downstream_couplings[layer].append(downstream)
                downstream_importances[layer].append(delta * downstream)
                native_couplings[layer].append(native)
                native_importances[layer].append(delta * native)
                direct_importances[layer].append(delta * direct)
                activation_deltas[layer].append(delta)

    layer_summaries: list[FFNCouplingLayerSummary] = []
    flattened_rms: list[torch.Tensor] = []
    offsets: list[tuple[int, int]] = []
    aggregate: dict[int, dict[str, torch.Tensor]] = {}
    total = 0
    for layer in layers:
        downstream_coupling = torch.stack(downstream_couplings[layer])
        downstream_importance = torch.stack(downstream_importances[layer])
        native_coupling = torch.stack(native_couplings[layer])
        native_importance = torch.stack(native_importances[layer])
        direct_importance = torch.stack(direct_importances[layer])
        delta = torch.stack(activation_deltas[layer])
        downstream_rms = downstream_importance.square().mean(dim=0).sqrt()
        native_rms = native_importance.square().mean(dim=0).sqrt()
        direct_rms = direct_importance.square().mean(dim=0).sqrt()
        downstream_mean = downstream_importance.mean(dim=0)
        native_mean = native_importance.mean(dim=0)
        downstream_consistency = _sign_consistency(downstream_importance)
        sign_agreement = (
            downstream_importance.sign() == direct_importance.sign()
        ).float().mean(dim=0)
        top_neuron = int(torch.argmax(downstream_rms).item())
        layer_summaries.append(
            FFNCouplingLayerSummary(
                layer=layer,
                downstream_rms_mass=float(downstream_rms.sum().item()),
                native_rms_mass=float(native_rms.sum().item()),
                direct_rms_mass=float(direct_rms.sum().item()),
                top_neuron=top_neuron,
            )
        )
        aggregate[layer] = {
            "delta_mean": delta.mean(dim=0),
            "direct": parent_tensors[f"coupling.layer_{layer}"].detach().float().cpu(),
            "direct_rms": direct_rms,
            "native_coupling_mean": native_coupling.mean(dim=0),
            "native_mean": native_mean,
            "native_rms": native_rms,
            "downstream_coupling_mean": downstream_coupling.mean(dim=0),
            "downstream_mean": downstream_mean,
            "downstream_rms": downstream_rms,
            "downstream_consistency": downstream_consistency,
            "sign_agreement": sign_agreement,
        }
        for name, value in aggregate[layer].items():
            tensors[f"{name}.layer_{layer}"] = value.contiguous()
        offsets.append((layer, total))
        total += downstream_rms.numel()
        flattened_rms.append(downstream_rms)

    flat = torch.cat(flattened_rms)
    chosen = min(spec.top_k, flat.numel())
    selected_flat = torch.argsort(flat, descending=True, stable=True)[:chosen].tolist()
    neurons: list[FFNCouplingNeuronScore] = []
    for rank, flat_index in enumerate(selected_flat, start=1):
        layer, offset = max(
            (item for item in offsets if item[1] <= flat_index), key=lambda item: item[1]
        )
        neuron = flat_index - offset
        values = aggregate[layer]
        neurons.append(
            FFNCouplingNeuronScore(
                rank=rank,
                layer=layer,
                neuron=neuron,
                activation_delta_mean=float(values["delta_mean"][neuron].item()),
                direct_coupling=float(values["direct"][neuron].item()),
                direct_importance_rms=float(values["direct_rms"][neuron].item()),
                native_coupling_mean=float(
                    values["native_coupling_mean"][neuron].item()
                ),
                native_importance_mean=float(values["native_mean"][neuron].item()),
                native_importance_rms=float(values["native_rms"][neuron].item()),
                downstream_coupling_mean=float(
                    values["downstream_coupling_mean"][neuron].item()
                ),
                downstream_importance_mean=float(
                    values["downstream_mean"][neuron].item()
                ),
                downstream_importance_rms=float(
                    values["downstream_rms"][neuron].item()
                ),
                downstream_sign_consistency=float(
                    values["downstream_consistency"][neuron].item()
                ),
                direct_downstream_sign_agreement=float(
                    values["sign_agreement"][neuron].item()
                ),
            )
        )

    summary = FFNCouplingRunSummary(
        science_hash=science_hash,
        parent_run_id=spec.parent_run_id,
        trajectory_run_id=spec.trajectory_run_id,
        model=asdict(engine.adapter.metadata),
        observable={
            "name": observable.name,
            "target_tokens": [item.text for item in observable.target],
            "control_tokens": [item.text for item in observable.control],
        },
        pair_count=len(pair_results),
        candidate_pair_ids=tuple(
            item.pair_id for item in pair_results if item.split == "discovery"
        ),
        logical_forward_passes=2 * len(pair_results),
        logical_backward_passes=2 * len(pair_results),
        methods=spec.methods,
        pairs=tuple(pair_results),
        layers=tuple(layer_summaries),
        neurons=tuple(neurons),
        total_neuron_count=total,
        warnings=(
            "Layer-aware coupling is a local first-order influence hypothesis; controlled intervention remains required.",
            "Candidate aggregation uses discovery pairs only; validation and held-out gradients are retained for separate diagnostics.",
        ),
    )
    return FFNCouplingComputation(summary=summary, tensors=tensors)
