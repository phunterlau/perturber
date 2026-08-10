from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .domain import LayerSummary, NeuronScore


@dataclass(frozen=True)
class RankedScores:
    layers: tuple[LayerSummary, ...]
    neurons: tuple[NeuronScore, ...]
    predicted_delta: float
    total_neuron_count: int
    importance_by_layer: tuple[torch.Tensor, ...]
    delta_by_layer: tuple[torch.Tensor, ...]


def logit_gap(
    logits: torch.Tensor,
    target_ids: tuple[int, ...],
    control_ids: tuple[int, ...],
) -> float:
    values = logits.detach().float().cpu()
    target = values[list(target_ids)].mean()
    control = values[list(control_ids)].mean()
    return float((target - control).item())


def behavioral_direction(
    output_weight: torch.Tensor,
    target_ids: tuple[int, ...],
    control_ids: tuple[int, ...],
) -> torch.Tensor:
    weight = output_weight.detach()
    target_index = torch.tensor(target_ids, device=weight.device, dtype=torch.long)
    control_index = torch.tensor(control_ids, device=weight.device, dtype=torch.long)
    target = weight.index_select(0, target_index).float().mean(dim=0)
    control = weight.index_select(0, control_index).float().mean(dim=0)
    return (target - control).cpu()


def ffn_skip_ratio(
    direction: torch.Tensor,
    layer_input: torch.Tensor,
    ffn_output: torch.Tensor,
    epsilon: float = 1e-8,
) -> float | None:
    d = direction.detach().float().cpu()
    denominator = abs(float(torch.dot(d, layer_input.detach().float().cpu()).item()))
    if denominator < epsilon:
        return None
    numerator = abs(float(torch.dot(d, ffn_output.detach().float().cpu()).item()))
    return numerator / denominator


def classify_circuit(ffn_skip_mean: float | None) -> str:
    if ffn_skip_mean is None or not math.isfinite(ffn_skip_mean):
        return "FFN signal concentration undetermined"
    if ffn_skip_mean > 0.3:
        return (
            "high FFN signal concentration (empirical opposition-compatible range; "
            "opposition and intervention evidence still required)"
        )
    if ffn_skip_mean < 0.2:
        return (
            "low FFN signal concentration (empirical routing/readout-compatible range; "
            "Mode 3 needs separate validation)"
        )
    return "intermediate FFN signal concentration (empirical transition range)"


def rank_neurons(
    original_activations: tuple[torch.Tensor, ...],
    perturbed_activations: tuple[torch.Tensor, ...],
    couplings: tuple[torch.Tensor, ...],
    top_k: int,
) -> RankedScores:
    if not (
        len(original_activations)
        == len(perturbed_activations)
        == len(couplings)
    ):
        raise ValueError("activation and coupling layer counts must match")
    if not original_activations:
        raise ValueError("at least one layer is required")

    layer_importances: list[torch.Tensor] = []
    layer_deltas: list[torch.Tensor] = []
    summaries: list[LayerSummary] = []
    offsets: list[int] = []
    layer_rank_maps: list[torch.Tensor] = []
    total = 0

    for layer, (original, perturbed, coupling) in enumerate(
        zip(original_activations, perturbed_activations, couplings, strict=True)
    ):
        original_cpu = original.detach().float().cpu().flatten()
        perturbed_cpu = perturbed.detach().float().cpu().flatten()
        coupling_cpu = coupling.detach().float().cpu().flatten()
        if not (
            original_cpu.shape == perturbed_cpu.shape == coupling_cpu.shape
        ):
            raise ValueError(f"layer {layer} activation/coupling shapes do not match")
        if original_cpu.numel() == 0:
            raise ValueError(f"layer {layer} must contain at least one neuron")
        if not all(
            torch.isfinite(value).all().item()
            for value in (original_cpu, perturbed_cpu, coupling_cpu)
        ):
            raise ValueError(f"layer {layer} contains non-finite values")

        delta = perturbed_cpu - original_cpu
        importance = coupling_cpu * delta
        absolute = importance.abs()
        absolute_sum = float(absolute.sum().item())
        top_count = min(10, importance.numel())
        top_mass = float(torch.topk(absolute, top_count).values.sum().item())
        top_index = int(torch.argmax(absolute).item())

        # Stable sorting makes equal-score rankings reproducible: the lower
        # neuron index wins a tie.
        order = torch.argsort(absolute, descending=True, stable=True)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel())
        layer_rank_maps.append(inverse + 1)

        summaries.append(
            LayerSummary(
                layer=layer,
                signed_sum=float(importance.sum().item()),
                absolute_sum=absolute_sum,
                positive_mass=float(importance.clamp_min(0).sum().item()),
                negative_mass=float((-importance.clamp_max(0)).sum().item()),
                top_10_share=(top_mass / absolute_sum if absolute_sum else 0.0),
                maximum_absolute=float(absolute[top_index].item()),
                top_neuron=top_index,
                activation_delta_norm=float(torch.linalg.vector_norm(delta).item()),
            )
        )
        offsets.append(total)
        total += importance.numel()
        layer_importances.append(importance)
        layer_deltas.append(delta)

    flattened = torch.cat(layer_importances)
    chosen = min(max(1, top_k), flattened.numel())
    flat_indices = torch.argsort(
        flattened.abs(), descending=True, stable=True
    )[:chosen].tolist()

    neurons: list[NeuronScore] = []
    for rank, flat_index in enumerate(flat_indices, start=1):
        layer = max(i for i, offset in enumerate(offsets) if offset <= flat_index)
        neuron = flat_index - offsets[layer]
        original = original_activations[layer].detach().float().cpu().flatten()
        perturbed = perturbed_activations[layer].detach().float().cpu().flatten()
        coupling = couplings[layer].detach().float().cpu().flatten()
        importance = layer_importances[layer]
        delta = layer_deltas[layer]
        neurons.append(
            NeuronScore(
                rank=rank,
                layer_rank=int(layer_rank_maps[layer][neuron].item()),
                layer=layer,
                neuron=neuron,
                coupling=float(coupling[neuron].item()),
                original_activation=float(original[neuron].item()),
                perturbed_activation=float(perturbed[neuron].item()),
                activation_delta=float(delta[neuron].item()),
                importance=float(importance[neuron].item()),
            )
        )

    return RankedScores(
        layers=tuple(summaries),
        neurons=tuple(neurons),
        predicted_delta=float(flattened.sum().item()),
        total_neuron_count=total,
        importance_by_layer=tuple(layer_importances),
        delta_by_layer=tuple(layer_deltas),
    )
