import pytest

from probing.domain import ObservableSpec
from probing.observables import ObservableResolutionError, parse_token_csv, resolve_observable
from helpers import FakeTokenizer


def test_resolve_observable_preserves_exact_token_strings() -> None:
    result = resolve_observable(
        FakeTokenizer(),
        ObservableSpec(
            name="agreement",
            target_tokens=("No",),
            control_tokens=("Yes",),
        ),
    )

    assert result.target_ids == (0,)
    assert result.control_ids == (1,)
    assert result.target[0].decoded == "No"


def test_resolve_observable_rejects_multi_token_entry() -> None:
    with pytest.raises(ObservableResolutionError, match="exactly one token"):
        resolve_observable(
            FakeTokenizer(),
            ObservableSpec(
                name="invalid",
                target_tokens=("multi",),
                control_tokens=("Yes",),
            ),
        )


def test_parse_token_csv() -> None:
    assert parse_token_csv(" No, Yes ,,Maybe ") == ("No", "Yes", "Maybe")
