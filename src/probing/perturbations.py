from __future__ import annotations

from difflib import SequenceMatcher
import string

from .contracts import (
    ExperimentSet,
    PerturbationCompilation,
    PerturbationDiagnostic,
    PerturbationTemplate,
    PromptPairInput,
)


def _required_fields(template: str) -> set[str]:
    return {
        field_name.split(".", 1)[0].split("[", 1)[0]
        for _literal, field_name, _format, _conversion in string.Formatter().parse(
            template
        )
        if field_name
    }


def compile_perturbations(template: PerturbationTemplate) -> PerturbationCompilation:
    required = _required_fields(template.original_template) | _required_fields(
        template.perturbed_template
    )
    pairs = []
    diagnostics = []
    global_warnings = []
    for case in template.cases:
        missing = sorted(required - set(case.values))
        if missing:
            raise ValueError(
                f"perturbation case {case.id!r} is missing template values {missing}"
            )
        original = template.original_template.format_map(case.values)
        perturbed = template.perturbed_template.format_map(case.values)
        if not original.strip() or not perturbed.strip():
            raise ValueError(f"perturbation case {case.id!r} rendered a blank prompt")
        original_words = original.split()
        perturbed_words = perturbed.split()
        matcher = SequenceMatcher(a=original_words, b=perturbed_words, autojunk=False)
        changed = sum(
            max(first_end - first_start, second_end - second_start)
            for operation, first_start, first_end, second_start, second_end in matcher.get_opcodes()
            if operation != "equal"
        )
        shared_fraction = matcher.ratio()
        length_ratio = len(perturbed_words) / max(1, len(original_words))
        warnings = []
        if original == perturbed:
            warnings.append("rendered original and perturbed prompts are identical")
        if shared_fraction < 0.5:
            warnings.append("less than half of the word sequence is shared")
        if not 0.8 <= length_ratio <= 1.25:
            warnings.append("prompt lengths differ by more than 25 percent")
        diagnostics.append(
            PerturbationDiagnostic(
                pair_id=case.id,
                changed_word_count=changed,
                shared_word_fraction=shared_fraction,
                length_ratio=length_ratio,
                warnings=tuple(warnings),
            )
        )
        pairs.append(
            PromptPairInput(
                id=case.id,
                original=original,
                perturbed=perturbed,
                split=case.split,
                metadata={
                    **case.metadata,
                    "perturbation_family": template.name,
                    "target_factor": template.target_factor,
                },
            )
        )
    if any(item.warnings for item in diagnostics):
        global_warnings.append(
            "Lexical diagnostics flag possible confounds but cannot verify semantic factor isolation."
        )
    return PerturbationCompilation(
        target_factor=template.target_factor,
        experiment_set=ExperimentSet(
            name=template.name,
            pairs=tuple(pairs),
            tags=template.tags,
        ),
        diagnostics=tuple(diagnostics),
        warnings=tuple(global_warnings),
    )


__all__ = ["compile_perturbations"]
