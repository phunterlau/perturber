from types import SimpleNamespace

import pytest
import torch

from probing.adapters.base import ActivationEdit, AttentionHeadEdit, ResidualEdit
from probing.adapters.qwen import QwenAdapter
from probing.domain import TokenizedPrompt


class BatchLike:
    def __init__(self, input_ids: torch.Tensor) -> None:
        self.input_ids = input_ids

    def __getitem__(self, key: str) -> torch.Tensor:
        assert key == "input_ids"
        return self.input_ids


class TemplateTokenizer:
    def apply_chat_template(self, *args, **kwargs):
        return BatchLike(torch.tensor([[1, 2]]))

    def decode(self, token_ids):
        return str(token_ids[0])


class RecordingTemplateTokenizer(TemplateTokenizer):
    def __init__(self) -> None:
        self.messages = None
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return BatchLike(torch.tensor([[3, 4]]))


class TinyMLP(torch.nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = torch.nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = torch.nn.Linear(intermediate, hidden, bias=False)

    def activation(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(self.gate_proj(hidden)) * self.up_proj(hidden)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.activation(hidden))


class TinyLayer(torch.nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.mlp = TinyMLP(hidden, intermediate)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.mlp(hidden)


class TinyInner(torch.nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [TinyLayer(hidden, intermediate), TinyLayer(hidden, intermediate)]
        )


class TinyQwen(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(7)
        self.config = SimpleNamespace(
            model_type="qwen3",
            _commit_hash="tiny",
        )
        self.embed = torch.nn.Embedding(8, 3)
        self.model = TinyInner(hidden=3, intermediate=5)
        self.lm_head = torch.nn.Linear(3, 8, bias=False)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False):
        hidden = self.embed(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return SimpleNamespace(logits=self.lm_head(hidden))


def test_qwen_adapter_captures_exact_post_swiglu_down_proj_input() -> None:
    model = TinyQwen()
    adapter = QwenAdapter(
        model_id="tiny/qwen3",
        model=model,
        tokenizer=object(),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    input_ids = torch.tensor([[1, 2]])
    tokenized = TokenizedPrompt(
        text="fixture",
        input_ids=(1, 2),
        decoded_tokens=("1", "2"),
    )

    hidden = model.embed(input_ids)
    expected_activations = []
    expected_last_input = None
    expected_last_output = None
    for index, layer in enumerate(model.model.layers):
        if index == len(model.model.layers) - 1:
            expected_last_input = hidden[0, -1].detach()
        activation = layer.mlp.activation(hidden)
        expected_activations.append(activation[0, -1].detach())
        output = layer.mlp.down_proj(activation)
        if index == len(model.model.layers) - 1:
            expected_last_output = output[0, -1].detach()
        hidden = hidden + output

    capture = adapter.forward_capture(input_ids, tokenized, -1)

    assert len(capture.activations) == 2
    for actual, expected in zip(
        capture.activations, expected_activations, strict=True
    ):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(capture.last_layer_input, expected_last_input)
    torch.testing.assert_close(capture.last_ffn_output, expected_last_output)
    assert all(not layer.mlp.down_proj._forward_pre_hooks for layer in model.model.layers)
    assert not model.model.layers[-1]._forward_pre_hooks
    assert not model.model.layers[-1].mlp.down_proj._forward_hooks


def test_prepare_input_accepts_transformers_batch_encoding_shape() -> None:
    adapter = QwenAdapter(
        model_id="tiny/qwen3",
        model=TinyQwen(),
        tokenizer=TemplateTokenizer(),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    input_ids, tokenized = adapter.prepare_input(
        "fixture",
        chat_template=True,
        enable_thinking=False,
    )

    torch.testing.assert_close(input_ids, torch.tensor([[1, 2]]))
    assert tokenized.input_ids == (1, 2)


def test_prepare_structured_chat_preserves_roles_and_tool_schema() -> None:
    tokenizer = RecordingTemplateTokenizer()
    adapter = QwenAdapter(
        model_id="tiny/qwen3",
        model=TinyQwen(),
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    messages = (
        {"role": "system", "content": "Use tools when needed."},
        {"role": "user", "content": "What is the weather?"},
    )
    tools = (
        {
            "type": "function",
            "function": {
                "name": "weather",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    )

    input_ids, tokenized = adapter.prepare_chat_input(
        messages,
        tools=tools,
        enable_thinking=False,
    )

    torch.testing.assert_close(input_ids, torch.tensor([[3, 4]]))
    assert tokenizer.messages == list(messages)
    assert tokenizer.kwargs["tools"] == list(tools)
    assert '"role": "system"' in tokenized.text


@pytest.mark.parametrize("position", [2, -3])
def test_capture_position_is_validated_before_forward(position) -> None:
    model = TinyQwen()
    adapter = QwenAdapter(
        model_id="tiny/qwen3",
        model=model,
        tokenizer=object(),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    input_ids = torch.tensor([[1, 2]])
    tokenized = TokenizedPrompt(
        text="fixture", input_ids=(1, 2), decoded_tokens=("1", "2")
    )

    with pytest.raises(ValueError, match="outside a prompt of 2 tokens"):
        adapter.forward_capture(input_ids, tokenized, position)

    assert all(not layer.mlp.down_proj._forward_pre_hooks for layer in model.model.layers)


def test_qwen_adapter_applies_non_mutating_neuron_ablation() -> None:
    model = TinyQwen()
    adapter = QwenAdapter(
        model_id="tiny/qwen3",
        model=model,
        tokenizer=TemplateTokenizer(),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    input_ids = torch.tensor([[1, 2]])
    tokenized = TokenizedPrompt(
        text="fixture", input_ids=(1, 2), decoded_tokens=("1", "2")
    )
    original_weight = model.model.layers[0].mlp.down_proj.weight.detach().clone()

    baseline = adapter.forward_capture(input_ids, tokenized, -1)
    edited = adapter.forward_intervened(
        input_ids,
        tokenized,
        -1,
        (
            ActivationEdit(
                layer=0,
                neurons=(0,),
                operation="scale",
                strength=0.0,
            ),
        ),
    )

    assert edited.activations[0][0] == 0
    assert not torch.equal(edited.logits, baseline.logits)
    torch.testing.assert_close(
        model.model.layers[0].mlp.down_proj.weight,
        original_weight,
    )
    assert all(not layer.mlp.down_proj._forward_pre_hooks for layer in model.model.layers)


def test_qwen_adapter_patch_mixes_only_selected_source_values() -> None:
    model = TinyQwen()
    adapter = QwenAdapter(
        model_id="tiny/qwen3",
        model=model,
        tokenizer=TemplateTokenizer(),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    input_ids = torch.tensor([[1, 2]])
    tokenized = TokenizedPrompt(
        text="fixture", input_ids=(1, 2), decoded_tokens=("1", "2")
    )
    baseline = adapter.forward_capture(input_ids, tokenized, -1)
    source = torch.full_like(baseline.activations[1], 4.0)

    edited = adapter.forward_intervened(
        input_ids,
        tokenized,
        -1,
        (
            ActivationEdit(
                layer=1,
                neurons=(1, 3),
                operation="mix",
                strength=0.25,
                source_values=source,
            ),
        ),
    )

    expected = baseline.activations[1].clone()
    indices = torch.tensor([1, 3])
    expected[indices] = baseline.activations[1][indices] * 0.75 + 1.0
    torch.testing.assert_close(edited.activations[1], expected)


def test_qwen_adapter_applies_norm_aware_residual_direction_without_weight_mutation() -> None:
    model = TinyQwen()
    adapter = QwenAdapter(
        model_id="tiny/qwen3",
        model=model,
        tokenizer=TemplateTokenizer(),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    input_ids = torch.tensor([[1, 2]])
    tokenized = TokenizedPrompt(
        text="fixture", input_ids=(1, 2), decoded_tokens=("1", "2")
    )
    baseline = adapter.forward_capture(input_ids, tokenized, -1)
    lm_head = model.lm_head.weight.detach().clone()

    edited = adapter.forward_residual_intervened(
        input_ids,
        tokenized,
        -1,
        (
            ResidualEdit(
                layer=0,
                direction=torch.tensor([1.0, -1.0, 0.5]),
                beta=0.2,
                normalization="residual_norm",
            ),
        ),
    )

    assert not torch.equal(edited.logits, baseline.logits)
    torch.testing.assert_close(model.lm_head.weight, lm_head)
    assert all(not layer._forward_hooks for layer in model.model.layers)


def _real_tiny_qwen_adapter() -> QwenAdapter:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(19)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
    )
    return QwenAdapter(
        model_id="tiny/real-qwen3",
        model=Qwen3ForCausalLM(config).eval(),
        tokenizer=TemplateTokenizer(),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_real_qwen_attention_capture_reconstructs_heads_with_gqa_mapping() -> None:
    adapter = _real_tiny_qwen_adapter()
    input_ids = torch.tensor([[1, 2, 3]])
    tokenized = TokenizedPrompt(
        text="fixture",
        input_ids=(1, 2, 3),
        decoded_tokens=("1", "2", "3"),
    )
    implementation = adapter.model.config._attn_implementation

    capture = adapter.forward_attention_capture(
        input_ids,
        tokenized,
        -1,
        include_attention_weights=True,
    )

    metadata = adapter.attention_metadata()
    assert metadata.output_head_count == 4
    assert metadata.key_value_head_count == 2
    assert [metadata.key_value_head(head) for head in range(4)] == [0, 0, 1, 1]
    assert capture.head_outputs[0].shape == (3, 4, 4)
    assert capture.attention_weights[0].shape == (4, 3)
    assert capture.values[0].shape == (3, 2, 4)
    for head in range(metadata.output_head_count):
        kv_head = metadata.key_value_head(head)
        reconstructed = (
            capture.attention_weights[0][head, :, None]
            * capture.values[0][:, kv_head, :]
        ).sum(dim=0)
        torch.testing.assert_close(
            reconstructed,
            capture.head_outputs[0][-1, head],
            atol=1e-6,
            rtol=1e-6,
        )
    assert adapter.model.config._attn_implementation == implementation
    for layer in adapter.model.model.layers:
        assert not layer.self_attn._forward_hooks
        assert not layer.self_attn.v_proj._forward_hooks
        assert not layer.self_attn.o_proj._forward_pre_hooks


def test_real_qwen_attention_head_edit_is_local_and_does_not_mutate_weights() -> None:
    adapter = _real_tiny_qwen_adapter()
    input_ids = torch.tensor([[1, 2, 3]])
    tokenized = TokenizedPrompt(
        text="fixture",
        input_ids=(1, 2, 3),
        decoded_tokens=("1", "2", "3"),
    )
    baseline = adapter.forward_attention_capture(input_ids, tokenized, -1)
    original_weight = (
        adapter.model.model.layers[0].self_attn.o_proj.weight.detach().clone()
    )

    edited = adapter.forward_attention_capture(
        input_ids,
        tokenized,
        -1,
        layers=(0,),
        edits=(
            AttentionHeadEdit(
                layer=0,
                heads=(1,),
                operation="scale",
                strength=0.0,
            ),
        ),
    )

    torch.testing.assert_close(edited.head_outputs[0][-1, 1], torch.zeros(4))
    torch.testing.assert_close(
        edited.head_outputs[0][:-1, 1], baseline.head_outputs[0][:-1, 1]
    )
    assert not torch.equal(edited.logits, baseline.logits)
    torch.testing.assert_close(
        adapter.model.model.layers[0].self_attn.o_proj.weight,
        original_weight,
    )
    assert not adapter.model.model.layers[0].self_attn.o_proj._forward_pre_hooks


def test_attention_output_coupling_matches_direct_projection() -> None:
    adapter = _real_tiny_qwen_adapter()
    direction = torch.linspace(-1.0, 1.0, 16)

    couplings = adapter.attention_output_couplings(direction)

    weight = adapter.model.model.layers[0].self_attn.o_proj.weight.detach()
    expected = torch.matmul(direction, weight).reshape(4, 4)
    torch.testing.assert_close(couplings[0], expected)
