from __future__ import annotations

from typing import Protocol

from .domain import ObservableSpec, ResolvedObservable, ResolvedToken


class TokenizerLike(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...

    def decode(self, token_ids: int | list[int]) -> str: ...


class ObservableResolutionError(ValueError):
    """Raised when an observable string is not one tokenizer token."""


def _resolve_set(
    tokenizer: TokenizerLike,
    values: tuple[str, ...],
    set_name: str,
) -> tuple[ResolvedToken, ...]:
    if not values:
        raise ObservableResolutionError(f"{set_name} token set may not be empty")

    resolved: list[ResolvedToken] = []
    seen: set[int] = set()
    for text in values:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ObservableResolutionError(
                f"{set_name} token {text!r} resolved to {len(token_ids)} tokens "
                f"{token_ids}; the MVP requires exactly one token per entry"
            )
        token_id = int(token_ids[0])
        if token_id in seen:
            continue
        seen.add(token_id)
        resolved.append(
            ResolvedToken(
                text=text,
                token_id=token_id,
                decoded=tokenizer.decode([token_id]),
            )
        )
    return tuple(resolved)


def resolve_observable(
    tokenizer: TokenizerLike,
    spec: ObservableSpec,
) -> ResolvedObservable:
    target = _resolve_set(tokenizer, spec.target_tokens, "target")
    control = _resolve_set(tokenizer, spec.control_tokens, "control")
    overlap = set(item.token_id for item in target) & set(
        item.token_id for item in control
    )
    if overlap:
        raise ObservableResolutionError(
            f"target and control token sets overlap at token IDs {sorted(overlap)}"
        )
    return ResolvedObservable(name=spec.name, target=target, control=control)


def parse_token_csv(value: str) -> tuple[str, ...]:
    tokens = tuple(part.strip() for part in value.split(",") if part.strip())
    if not tokens:
        raise ObservableResolutionError("token list may not be empty")
    return tokens
