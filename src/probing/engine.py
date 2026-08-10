from __future__ import annotations

from statistics import fmean
import time
from datetime import datetime, timezone
from dataclasses import dataclass

import torch

from .adapters import ModelAdapter, QwenAdapter
from .domain import ALGORITHM_VERSION, NextTokenPrediction, ProbeResult, ProbeSpec
from .observables import resolve_observable
from .scoring import (
    classify_circuit,
    ffn_skip_ratio,
    logit_gap,
    rank_neurons,
)


@dataclass(frozen=True)
class ProbeAnalysis:
    result: ProbeResult
    original_logits: torch.Tensor
    perturbed_logits: torch.Tensor
    behavioral_direction: torch.Tensor
    importance_by_layer: tuple[torch.Tensor, ...]
    coupling_by_layer: tuple[torch.Tensor, ...]
    original_activation_by_layer: tuple[torch.Tensor, ...]
    perturbed_activation_by_layer: tuple[torch.Tensor, ...]


class ProbeEngine:
    def __init__(self, adapter: ModelAdapter) -> None:
        self.adapter = adapter

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        revision: str | None = None,
        device: str = "auto",
        dtype: str = "auto",
        cache_dir: str | None = None,
        local_files_only: bool = False,
        model_path: str | None = None,
    ) -> "ProbeEngine":
        adapter = QwenAdapter.from_pretrained(
            model_id,
            revision=revision,
            device=device,
            dtype=dtype,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            model_path=model_path,
        )
        return cls(adapter)

    def analyze(self, spec: ProbeSpec) -> ProbeResult:
        return self.analyze_details(spec).result

    def analyze_details(self, spec: ProbeSpec) -> ProbeAnalysis:
        started = time.perf_counter()
        warnings: list[str] = [
            "Exploratory result from one prompt pair; causal claims require intervention and replication."
        ]
        if spec.model_id != self.adapter.metadata.model_id:
            raise ValueError(
                f"spec model {spec.model_id!r} does not match loaded model "
                f"{self.adapter.metadata.model_id!r}"
            )

        observable = resolve_observable(self.adapter.tokenizer, spec.observable)
        original_ids, original_tokens = self.adapter.prepare_prompt(
            text=spec.pair.original,
            messages=spec.pair.original_messages,
            tools=spec.pair.tools,
            chat_template=spec.chat_template,
            enable_thinking=spec.enable_thinking,
        )
        perturbed_ids, perturbed_tokens = self.adapter.prepare_prompt(
            text=spec.pair.perturbed,
            messages=spec.pair.perturbed_messages,
            tools=spec.pair.tools,
            chat_template=spec.chat_template,
            enable_thinking=spec.enable_thinking,
        )

        original = self.adapter.forward_capture(
            original_ids,
            original_tokens,
            spec.capture_position,
        )
        perturbed = self.adapter.forward_capture(
            perturbed_ids,
            perturbed_tokens,
            spec.capture_position,
        )

        original_gap = logit_gap(
            original.logits,
            observable.target_ids,
            observable.control_ids,
        )
        perturbed_gap = logit_gap(
            perturbed.logits,
            observable.target_ids,
            observable.control_ids,
        )
        measured_delta = perturbed_gap - original_gap

        def prediction(logits: torch.Tensor) -> NextTokenPrediction:
            token_id = int(torch.argmax(logits).item())
            probability = float(torch.softmax(logits.float(), dim=0)[token_id].item())
            return NextTokenPrediction(
                token_id=token_id,
                decoded=self.adapter.tokenizer.decode([token_id]),
                logit=float(logits[token_id].item()),
                probability=probability,
            )

        direction = self.adapter.behavioral_direction(observable)
        couplings = self.adapter.structural_couplings(direction)
        ranked = rank_neurons(
            original.activations,
            perturbed.activations,
            couplings,
            spec.top_k,
        )

        original_ratio = ffn_skip_ratio(
            direction,
            original.last_layer_input,
            original.last_ffn_output,
        )
        perturbed_ratio = ffn_skip_ratio(
            direction,
            perturbed.last_layer_input,
            perturbed.last_ffn_output,
        )
        finite_ratios = [
            value for value in (original_ratio, perturbed_ratio) if value is not None
        ]
        mean_ratio = fmean(finite_ratios) if finite_ratios else None
        if mean_ratio is None:
            warnings.append(
                "FFN/Skip is undefined because the projected skip denominator is near zero."
            )

        if abs(measured_delta) > 1e-6:
            relative_error = abs(ranked.predicted_delta - measured_delta) / abs(
                measured_delta
            )
            if relative_error > 0.5:
                warnings.append(
                    "The additive neuron prediction differs from measured delta F by more "
                    "than 50%; treat the ranking as a weak hypothesis for this pair."
                )
        else:
            warnings.append(
                "The prompt pair produced almost no observable movement; revise the pair or token sets."
            )

        elapsed = time.perf_counter() - started
        result = ProbeResult(
            algorithm_version=ALGORITHM_VERSION,
            created_at=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=elapsed,
            spec=spec,
            model=self.adapter.metadata,
            observable=observable,
            original=original.tokenized,
            perturbed=perturbed.tokenized,
            original_prediction=prediction(original.logits),
            perturbed_prediction=prediction(perturbed.logits),
            original_gap=original_gap,
            perturbed_gap=perturbed_gap,
            measured_delta=measured_delta,
            predicted_delta=ranked.predicted_delta,
            ffn_skip_original=original_ratio,
            ffn_skip_perturbed=perturbed_ratio,
            ffn_skip_mean=mean_ratio,
            circuit_regime=classify_circuit(mean_ratio),
            layers=ranked.layers,
            neurons=ranked.neurons,
            total_neuron_count=ranked.total_neuron_count,
            warnings=tuple(warnings),
            runtime={
                "logical_forward_passes": 2,
                "behavioral_direction_norm": float(direction.norm().item()),
            },
        )
        return ProbeAnalysis(
            result=result,
            original_logits=original.logits.detach().float().cpu(),
            perturbed_logits=perturbed.logits.detach().float().cpu(),
            behavioral_direction=direction.detach().float().cpu(),
            importance_by_layer=ranked.importance_by_layer,
            coupling_by_layer=tuple(item.detach().float().cpu() for item in couplings),
            original_activation_by_layer=tuple(
                item.detach().float().cpu() for item in original.activations
            ),
            perturbed_activation_by_layer=tuple(
                item.detach().float().cpu() for item in perturbed.activations
            ),
        )
