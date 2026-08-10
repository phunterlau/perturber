from __future__ import annotations

from dataclasses import asdict
import random
from statistics import fmean

import torch

from .adapters import AttentionHeadEdit
from .attention import _eligible_parent_pairs
from .contracts import (
    AttentionHeadInterventionRunSummary,
    AttentionHeadRankRunSummary,
    AttentionHeadReference,
    AttentionPathObservation,
    AttentionTokenEdge,
    AttentionTraceRunSummary,
    AttentionTraceSpec,
    ClaimRecord,
    RankRunSummary,
    RankSpec,
    TokenAlignmentRequest,
)
from .domain import ObservableSpec
from .engine import ProbeEngine
from .observables import resolve_observable
from .prompting import prepare_pair_condition
from .scoring import logit_gap


def _validate_heads(
    *,
    heads: tuple[AttentionHeadReference, ...],
    layer_count: int,
    output_head_count: int,
    label: str,
) -> None:
    for item in heads:
        if item.layer >= layer_count or item.head >= output_head_count:
            raise ValueError(
                f"{label} attention head L{item.layer}:H{item.head} is out of range"
            )


def _alignment_positions(
    *,
    alignment: TokenAlignmentRequest,
    original_length: int,
    perturbed_length: int,
    operation: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if alignment.mode == "identity":
        if original_length != perturbed_length:
            raise ValueError(
                f"identity alignment requires equal token counts, got "
                f"{original_length} and {perturbed_length}"
            )
        original_to_perturbed = tuple(
            (position, position) for position in range(original_length)
        )
    else:
        original_to_perturbed = tuple(
            (item.original, item.perturbed) for item in alignment.positions
        )
        original_positions = {item[0] for item in original_to_perturbed}
        perturbed_positions = {item[1] for item in original_to_perturbed}
        if original_positions != set(range(original_length)):
            raise ValueError(
                "explicit alignment must map every original token position exactly once"
            )
        if perturbed_positions != set(range(perturbed_length)):
            raise ValueError(
                "explicit alignment must map every perturbed token position exactly once"
            )
    if operation == "patch":
        # Source is perturbed; target is original.
        ordered = sorted(original_to_perturbed, key=lambda item: item[0])
        target_positions = tuple(item[0] for item in ordered)
        source_positions = tuple(item[1] for item in ordered)
    else:
        # Source is original; target is perturbed.
        ordered = sorted(original_to_perturbed, key=lambda item: item[1])
        target_positions = tuple(item[1] for item in ordered)
        source_positions = tuple(item[0] for item in ordered)
    return source_positions, target_positions


def _trace_pairs(
    *,
    rank_spec: RankSpec,
    rank_summary: RankRunSummary,
    spec: AttentionTraceSpec,
    qualification_statuses: dict[str, str] | None = None,
):
    # Weak pairs are excluded by default. Explicit inclusion keeps the run
    # available for method development while claim language stays exploratory.
    return _eligible_parent_pairs(
        parent_spec=rank_spec,
        parent_summary=rank_summary,
        pair_ids=spec.pair_ids,
        include_weak_pairs=spec.include_weak_pairs,
        qualification_statuses=qualification_statuses,
        stage="attention trace",
    )


def attention_trace_plan_counts(
    *,
    rank_spec: RankSpec,
    rank_summary: RankRunSummary,
    attention_summary: AttentionHeadRankRunSummary,
    intervention_summary: AttentionHeadInterventionRunSummary | None,
    spec: AttentionTraceSpec,
    qualification_statuses: dict[str, str] | None = None,
) -> tuple[int, int]:
    pairs = _trace_pairs(
        rank_spec=rank_spec,
        rank_summary=rank_summary,
        spec=spec,
        qualification_statuses=qualification_statuses,
    )
    if spec.trace_kind == "token_edges":
        return len(pairs), 2 * len(pairs)
    if intervention_summary is None:
        raise ValueError("head path tracing requires an attention intervention summary")
    tested = {
        (item.layer, item.head) for item in intervention_summary.selected_heads
    }
    requested = {
        (item.layer, item.head) for item in spec.senders + spec.receivers
    }
    untested = sorted(requested - tested)
    if untested:
        raise ValueError(
            "head path endpoints must be present in the parent intervention's "
            f"selected population; untested={untested}"
        )
    if spec.controls.samples:
        selected_per_layer = {
            layer: sum(item_layer == layer for item_layer, _head in tested)
            for layer in {item.layer for item in spec.senders + spec.receivers}
        }
        for sender in spec.senders:
            for receiver in spec.receivers:
                available = (
                    attention_summary.output_head_count
                    - selected_per_layer.get(sender.layer, 0)
                ) * (
                    attention_summary.output_head_count
                    - selected_per_layer.get(receiver.layer, 0)
                )
                if spec.controls.samples > available:
                    raise ValueError(
                        "too few unique same-layer paths outside the parent "
                        "intervention population for matched controls: "
                        f"requested={spec.controls.samples} available={available}"
                    )
    alignment_ids = {item.pair_id for item in spec.alignments}
    pair_ids = {pair.id for _, pair, _ in pairs}
    if alignment_ids != pair_ids:
        raise ValueError(
            "head path alignments must match selected pair IDs exactly; "
            f"missing={sorted(pair_ids - alignment_ids)} "
            f"extra={sorted(alignment_ids - pair_ids)}"
        )
    path_count = len(spec.senders) * len(spec.receivers)
    arm_count = path_count * (1 + spec.controls.samples)
    # One source-condition cache plus an intermediate sender patch and a final
    # receiver-only patch for every selected or random-control path.
    return len(pairs), len(pairs) * (1 + 2 * arm_count)


def _token_edges(
    *,
    engine: ProbeEngine,
    rank_spec: RankSpec,
    rank_summary: RankRunSummary,
    spec: AttentionTraceSpec,
    pairs,
) -> tuple[tuple[AttentionTokenEdge, ...], int]:
    observable = resolve_observable(
        engine.adapter.tokenizer,
        ObservableSpec(
            name=rank_spec.observable.name,
            target_tokens=rank_spec.observable.target_tokens,
            control_tokens=rank_spec.observable.control_tokens,
        ),
    )
    direction = engine.adapter.behavioral_direction(observable)
    couplings = engine.adapter.attention_output_couplings(direction)
    metadata = engine.adapter.attention_metadata()
    layers = tuple(sorted({item.layer for item in spec.heads}))
    requested = {(item.layer, item.head) for item in spec.heads}
    edges: list[AttentionTokenEdge] = []
    model_calls = 0
    for _parent_index, pair, _summary in pairs:
        for condition in ("original", "perturbed"):
            input_ids, tokenized = prepare_pair_condition(
                engine.adapter,
                pair=pair,
                model=rank_spec.model,
                condition=condition,
            )
            capture = engine.adapter.forward_attention_capture(
                input_ids,
                tokenized,
                rank_spec.capture.position,
                layers=layers,
                include_attention_weights=True,
            )
            model_calls += 1
            for layer in layers:
                weights = capture.attention_weights[layer]
                values = capture.values[layer]
                outputs = capture.head_outputs[layer]
                if weights is None or values is None or outputs is None:
                    raise RuntimeError(
                        f"missing eager attention tensors for layer {layer}"
                    )
                for head in range(metadata.output_head_count):
                    if (layer, head) not in requested:
                        continue
                    key_value_head = metadata.key_value_head(head)
                    contributions = (
                        weights[head, :, None] * values[:, key_value_head, :]
                    )
                    reconstructed = contributions.sum(dim=0)
                    actual = outputs[-1, head]
                    if not torch.allclose(
                        reconstructed, actual, atol=5e-3, rtol=5e-3
                    ):
                        maximum = float((reconstructed - actual).abs().max().item())
                        raise RuntimeError(
                            "token-edge contributions failed head-output reconstruction "
                            f"for L{layer}:H{head}; max_abs_error={maximum:.6g}"
                        )
                    direct = torch.matmul(
                        contributions.float(), couplings[layer][head].float()
                    )
                    norms = torch.linalg.vector_norm(contributions.float(), dim=-1)
                    for position in range(len(tokenized.input_ids)):
                        edges.append(
                            AttentionTokenEdge(
                                pair_id=pair.id,
                                condition=condition,
                                layer=layer,
                                head=head,
                                key_value_head=key_value_head,
                                source_position=position,
                                source_token_id=tokenized.input_ids[position],
                                source_token=tokenized.decoded_tokens[position],
                                attention_weight=float(weights[head, position].item()),
                                direct_effect=float(direct[position].item()),
                                output_norm=float(norms[position].item()),
                            )
                        )
    edges.sort(
        key=lambda item: (
            -abs(item.direct_effect),
            item.pair_id,
            item.condition,
            item.layer,
            item.head,
            item.source_position,
        )
    )
    return tuple(edges[: spec.max_token_edges]), model_calls


def _random_path_candidates(
    *,
    sender: AttentionHeadReference,
    receiver: AttentionHeadReference,
    excluded: set[tuple[int, int]],
    output_head_count: int,
) -> tuple[tuple[AttentionHeadReference, AttentionHeadReference], ...]:
    sender_candidates = [
        head
        for head in range(output_head_count)
        if (sender.layer, head) not in excluded
    ]
    receiver_candidates = [
        head
        for head in range(output_head_count)
        if (receiver.layer, head) not in excluded
    ]
    candidates = [
        (
            AttentionHeadReference(layer=sender.layer, head=sender_head),
            AttentionHeadReference(layer=receiver.layer, head=receiver_head),
        )
        for sender_head in sender_candidates
        for receiver_head in receiver_candidates
    ]
    if not candidates:
        raise ValueError("too few same-layer heads for matched path controls")
    return tuple(candidates)


def _head_paths(
    *,
    engine: ProbeEngine,
    rank_spec: RankSpec,
    rank_summary: RankRunSummary,
    intervention_summary: AttentionHeadInterventionRunSummary,
    spec: AttentionTraceSpec,
    pairs,
) -> tuple[tuple[AttentionPathObservation, ...], int]:
    observable = resolve_observable(
        engine.adapter.tokenizer,
        ObservableSpec(
            name=rank_spec.observable.name,
            target_tokens=rank_spec.observable.target_tokens,
            control_tokens=rank_spec.observable.control_tokens,
        ),
    )
    metadata = engine.adapter.attention_metadata()
    tested = {
        (item.layer, item.head) for item in intervention_summary.selected_heads
    }
    requested = {
        (item.layer, item.head) for item in spec.senders + spec.receivers
    }
    untested = sorted(requested - tested)
    if untested:
        raise ValueError(
            "head path endpoints must be present in the parent intervention's "
            f"selected population; untested={untested}"
        )
    alignments = {item.pair_id: item for item in spec.alignments}
    selected_pair_ids = {pair.id for _, pair, _ in pairs}
    if set(alignments) != selected_pair_ids:
        raise ValueError(
            "head path alignments must match selected pair IDs exactly; "
            f"missing={sorted(selected_pair_ids - set(alignments))} "
            f"extra={sorted(set(alignments) - selected_pair_ids)}"
        )
    selected_paths = tuple(
        (sender, receiver)
        for sender in spec.senders
        for receiver in spec.receivers
    )
    # Controls must come from outside the entire causally selected population,
    # not merely outside the one sender/receiver pair currently being tested.
    excluded = tested
    rng = random.Random(spec.execution.seed)
    observations: list[AttentionPathObservation] = []
    model_calls = 0

    for _parent_index, pair, pair_summary in pairs:
        original_ids, original_tokens = prepare_pair_condition(
            engine.adapter,
            pair=pair,
            model=rank_spec.model,
            condition="original",
        )
        perturbed_ids, perturbed_tokens = prepare_pair_condition(
            engine.adapter,
            pair=pair,
            model=rank_spec.model,
            condition="perturbed",
        )
        source_ids, source_tokens = (
            (perturbed_ids, perturbed_tokens)
            if spec.operation == "patch"
            else (original_ids, original_tokens)
        )
        target_ids, target_tokens = (
            (original_ids, original_tokens)
            if spec.operation == "patch"
            else (perturbed_ids, perturbed_tokens)
        )
        source_positions, target_positions = _alignment_positions(
            alignment=alignments[pair.id],
            original_length=len(original_tokens.input_ids),
            perturbed_length=len(perturbed_tokens.input_ids),
            operation=spec.operation,
        )
        source_layers = tuple(sorted({item.layer for item in spec.senders}))
        source_capture = engine.adapter.forward_attention_capture(
            source_ids,
            source_tokens,
            rank_spec.capture.position,
            layers=source_layers,
        )
        model_calls += 1
        baseline_gap = (
            pair_summary.original_gap
            if spec.operation == "patch"
            else pair_summary.perturbed_gap
        )
        source_gap = (
            pair_summary.perturbed_gap
            if spec.operation == "patch"
            else pair_summary.original_gap
        )

        path_arms: list[
            tuple[
                str,
                int | None,
                AttentionHeadReference,
                AttentionHeadReference,
            ]
        ] = []
        for sender, receiver in selected_paths:
            path_arms.append(("selected_path", None, sender, receiver))
            candidates = _random_path_candidates(
                sender=sender,
                receiver=receiver,
                excluded=excluded,
                output_head_count=metadata.output_head_count,
            )
            if spec.controls.samples > len(candidates):
                raise ValueError(
                    "too few unique same-layer paths for matched controls: "
                    f"requested={spec.controls.samples} available={len(candidates)}"
                )
            sampled = rng.sample(candidates, spec.controls.samples)
            for sample, (random_sender, random_receiver) in enumerate(sampled):
                path_arms.append(
                    (
                        "matched_random_path",
                        sample,
                        random_sender,
                        random_receiver,
                    )
                )

        for arm, sample, sender, receiver in path_arms:
            source_outputs = source_capture.head_outputs[sender.layer]
            if source_outputs is None:
                raise RuntimeError(
                    f"missing source attention output for layer {sender.layer}"
                )
            source_values = torch.stack(
                [source_outputs[position, sender.head] for position in source_positions]
            ).unsqueeze(1)
            sender_edit = AttentionHeadEdit(
                layer=sender.layer,
                heads=(sender.head,),
                operation="mix",
                strength=1.0,
                positions=target_positions,
                source_values=source_values,
            )
            intermediate = engine.adapter.forward_attention_capture(
                target_ids,
                target_tokens,
                rank_spec.capture.position,
                layers=(receiver.layer,),
                edits=(sender_edit,),
            )
            model_calls += 1
            receiver_outputs = intermediate.head_outputs[receiver.layer]
            if receiver_outputs is None:
                raise RuntimeError(
                    f"missing receiver attention output for layer {receiver.layer}"
                )
            receiver_value = receiver_outputs[-1, receiver.head].reshape(
                1, 1, metadata.head_dim
            )
            receiver_edit = AttentionHeadEdit(
                layer=receiver.layer,
                heads=(receiver.head,),
                operation="mix",
                strength=1.0,
                positions=(-1,),
                source_values=receiver_value,
            )
            final = engine.adapter.forward_attention_capture(
                target_ids,
                target_tokens,
                rank_spec.capture.position,
                layers=(receiver.layer,),
                edits=(receiver_edit,),
            )
            model_calls += 1
            sender_gap = logit_gap(
                intermediate.logits, observable.target_ids, observable.control_ids
            )
            path_gap = logit_gap(
                final.logits, observable.target_ids, observable.control_ids
            )
            progress = None
            if abs(source_gap - baseline_gap) > 1e-8:
                progress = (path_gap - baseline_gap) / (
                    source_gap - baseline_gap
                )
            observations.append(
                AttentionPathObservation(
                    pair_id=pair.id,
                    split=pair.split,
                    arm=arm,
                    control_sample=sample,
                    operation=spec.operation,
                    sender=sender,
                    receiver=receiver,
                    baseline_gap=baseline_gap,
                    source_gap=source_gap,
                    sender_patched_gap=sender_gap,
                    path_patched_gap=path_gap,
                    sender_total_effect=sender_gap - baseline_gap,
                    path_specific_effect=path_gap - baseline_gap,
                    normalized_source_progress=progress,
                    alignment_mode=alignments[pair.id].mode,
                )
            )
    return tuple(observations), model_calls


def run_attention_trace(
    *,
    engine: ProbeEngine,
    rank_spec: RankSpec,
    rank_summary: RankRunSummary,
    attention_summary: AttentionHeadRankRunSummary,
    intervention_summary: AttentionHeadInterventionRunSummary | None,
    spec: AttentionTraceSpec,
    science_hash: str,
    qualification_statuses: dict[str, str] | None = None,
) -> AttentionTraceRunSummary:
    metadata = engine.adapter.attention_metadata()
    _validate_heads(
        heads=spec.heads + spec.senders + spec.receivers,
        layer_count=metadata.layer_count,
        output_head_count=metadata.output_head_count,
        label="requested",
    )
    pairs = _trace_pairs(
        rank_spec=rank_spec,
        rank_summary=rank_summary,
        spec=spec,
        qualification_statuses=qualification_statuses,
    )
    selected_statuses = [
        (
            qualification_statuses.get(pair.id)
            if qualification_statuses is not None
            else (
                pair_summary.qualification.status
                if pair_summary.qualification is not None
                else None
            )
        )
        for _parent_index, pair, pair_summary in pairs
    ]
    weak_evidence = any(
        status not in {None, "informative"} for status in selected_statuses
    )
    if spec.trace_kind == "token_edges":
        token_edges, model_calls = _token_edges(
            engine=engine,
            rank_spec=rank_spec,
            rank_summary=rank_summary,
            spec=spec,
            pairs=pairs,
        )
        paths: tuple[AttentionPathObservation, ...] = ()
        evidence_stage = "attention_hypothesis"
        warning_values = [
            "Token routes use eager attention weights multiplied by value vectors and output coupling.",
            "Attention weights and token-edge direct effects are observational, not causal path evidence.",
            "Only routes into the first-token decision position are included.",
        ]
        if weak_evidence:
            warning_values.append(
                "At least one included pair did not pass the informative-observable gate."
            )
        warnings = tuple(warning_values)
        claim = ClaimRecord(
            claim_id="attention-token-route-hypothesis",
            claim_type="attention_routing",
            status="exploratory",
            statement="Token-to-head contributions identify routes for path intervention.",
            limitations=warnings,
        )
    else:
        if intervention_summary is None:
            raise ValueError("head path tracing requires an intervention summary")
        paths, model_calls = _head_paths(
            engine=engine,
            rank_spec=rank_spec,
            rank_summary=rank_summary,
            intervention_summary=intervention_summary,
            spec=spec,
            pairs=pairs,
        )
        token_edges = ()
        evidence_stage = "attention_causal_paths"
        selected = [
            abs(item.path_specific_effect)
            for item in paths
            if item.arm == "selected_path"
        ]
        controls = [
            abs(item.path_specific_effect)
            for item in paths
            if item.arm == "matched_random_path"
        ]
        supported = bool(
            selected
            and (
                not controls
                or fmean(selected) > fmean(controls)
            )
            and fmean(selected) > 0
        )
        warning_values = [
            "Path effects are two-stage receiver-mediated patches under exact token alignment.",
            "Effects are local to the declared prompts, endpoint heads, and first-token observable.",
            "Unique matched random paths exclude the parent intervention's selected head population.",
            "A patched sender-to-receiver route need not be a unique computational path.",
        ]
        if weak_evidence:
            warning_values.append(
                "At least one included pair did not pass the informative-observable gate; path evidence is exploratory."
            )
        warnings = tuple(warning_values)
        claim = ClaimRecord(
            claim_id="sender-receiver-path-effect",
            claim_type="causal_path",
            status=(
                "exploratory"
                if supported and weak_evidence
                else "supported"
                if supported
                else "not_supported"
            ),
            statement=(
                "Two-stage path patching tested whether sender effects reached the observable through a receiver head."
            ),
            limitations=warnings,
        )
    return AttentionTraceRunSummary(
        science_hash=science_hash,
        parent_run_id=spec.parent_run_id,
        rank_run_id=attention_summary.parent_run_id,
        parent_intervention_run_id=spec.parent_intervention_run_id,
        model=asdict(engine.adapter.metadata),
        observable=rank_summary.observable,
        trace_kind=spec.trace_kind,
        pairs=tuple(pair.id for _, pair, _ in pairs),
        token_edges=token_edges,
        paths=paths,
        logical_forward_passes=model_calls,
        evidence_stage=evidence_stage,
        claims=(claim,),
        warnings=warnings,
    )


__all__ = ["attention_trace_plan_counts", "run_attention_trace"]
