from __future__ import annotations

from contextlib import ExitStack
import json
import os
from pathlib import Path
from typing import Any

import torch

from ..domain import ModelMetadata, ResolvedObservable, TokenizedPrompt
from ..scoring import behavioral_direction
from .base import (
    ActivationEdit,
    AttentionForwardCapture,
    AttentionHeadEdit,
    AttentionMetadata,
    ForwardCapture,
    GeneratedSequence,
    ModelAdapter,
    ResidualEdit,
)


SUPPORTED_MODEL_TYPES = {"qwen3"}


def _select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _select_dtype(requested: str, device: torch.device) -> torch.dtype:
    choices = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if requested != "auto":
        try:
            return choices[requested]
        except KeyError as exc:
            raise ValueError(f"unsupported dtype {requested!r}") from exc
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


class QwenAdapter(ModelAdapter):
    """Dense Qwen3 adapter capturing each down projection's input."""

    def __init__(
        self,
        *,
        model_id: str,
        model: torch.nn.Module,
        tokenizer: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.model_id = model_id
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype

        model_type = str(getattr(model.config, "model_type", "unknown"))
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"QwenAdapter supports {sorted(SUPPORTED_MODEL_TYPES)}, got {model_type!r}"
            )
        try:
            self.layers = tuple(model.model.layers)
        except AttributeError as exc:
            raise ValueError("model does not expose model.layers") from exc
        if not self.layers:
            raise ValueError("model has no decoder layers")
        for index, layer in enumerate(self.layers):
            if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "down_proj"):
                raise ValueError(f"layer {index} does not expose mlp.down_proj")

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
    ) -> "QwenAdapter":
        if cache_dir is not None:
            cache_root = Path(cache_dir).expanduser().resolve()
            cache_root.mkdir(parents=True, exist_ok=True)
            # Hugging Face's optional Xet client keeps a separate cache and log
            # location. Bind every cache to the explicit project directory and
            # use the ordinary HTTP path for a predictable local-Mac MVP.
            os.environ["HF_HOME"] = str(cache_root)
            os.environ["HF_HUB_CACHE"] = str(cache_root / "hub")
            os.environ["HF_XET_CACHE"] = str(cache_root / "xet")
            os.environ["HF_HUB_DISABLE_XET"] = "1"

        from transformers import AutoModelForCausalLM, AutoTokenizer

        selected_device = _select_device(device)
        selected_dtype = _select_dtype(dtype, selected_device)
        source = model_path or model_id
        source_revision = None if model_path is not None else revision
        tokenizer = AutoTokenizer.from_pretrained(
            source,
            revision=source_revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        model = AutoModelForCausalLM.from_pretrained(
            source,
            revision=source_revision,
            dtype=selected_dtype,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            low_cpu_mem_usage=True,
        )
        model = model.to(selected_device)
        model.eval()
        return cls(
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            device=selected_device,
            dtype=selected_dtype,
        )

    @property
    def metadata(self) -> ModelMetadata:
        parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        return ModelMetadata(
            model_id=self.model_id,
            resolved_revision=getattr(self.model.config, "_commit_hash", None),
            model_type=str(self.model.config.model_type),
            adapter="qwen3-dense-swiglu-v1",
            device=str(self.device),
            dtype=str(self.dtype).removeprefix("torch."),
            parameter_count=parameter_count,
            layer_count=len(self.layers),
        )

    def prepare_input(
        self,
        text: str,
        *,
        chat_template: bool,
        enable_thinking: bool,
    ) -> tuple[torch.Tensor, TokenizedPrompt]:
        if chat_template:
            template_result = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                return_tensors="pt",
            )
            if not torch.is_tensor(template_result):
                input_ids = template_result["input_ids"]
            else:
                input_ids = template_result
        else:
            encoded = self.tokenizer(text, return_tensors="pt")
            input_ids = encoded["input_ids"]

        return self._tokenized_input(input_ids, text=text)

    def prepare_chat_input(
        self,
        messages: tuple[dict[str, Any], ...],
        *,
        tools: tuple[dict[str, Any], ...],
        enable_thinking: bool,
    ) -> tuple[torch.Tensor, TokenizedPrompt]:
        kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
            "enable_thinking": enable_thinking,
            "return_tensors": "pt",
        }
        if tools:
            kwargs["tools"] = list(tools)
        template_result = self.tokenizer.apply_chat_template(
            list(messages),
            **kwargs,
        )
        input_ids = (
            template_result
            if torch.is_tensor(template_result)
            else template_result["input_ids"]
        )
        text = json.dumps(
            {"messages": messages, "tools": tools},
            ensure_ascii=False,
            sort_keys=True,
        )
        return self._tokenized_input(input_ids, text=text)

    def _tokenized_input(
        self, input_ids: torch.Tensor, *, text: str
    ) -> tuple[torch.Tensor, TokenizedPrompt]:
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        ids = tuple(int(value) for value in input_ids[0].tolist())
        decoded = tuple(self.tokenizer.decode([token_id]) for token_id in ids)
        tokenized = TokenizedPrompt(text=text, input_ids=ids, decoded_tokens=decoded)
        return input_ids, tokenized

    def forward_capture(
        self,
        input_ids: torch.Tensor,
        tokenized: TokenizedPrompt,
        capture_position: int,
    ) -> ForwardCapture:
        return self._forward_capture(
            input_ids,
            tokenized,
            capture_position,
            edits=(),
            residual_edits=(),
        )

    def forward_intervened(
        self,
        input_ids: torch.Tensor,
        tokenized: TokenizedPrompt,
        capture_position: int,
        edits: tuple[ActivationEdit, ...],
    ) -> ForwardCapture:
        return self._forward_capture(
            input_ids,
            tokenized,
            capture_position,
            edits=edits,
            residual_edits=(),
        )

    def forward_residual_intervened(
        self,
        input_ids: torch.Tensor,
        tokenized: TokenizedPrompt,
        capture_position: int,
        edits: tuple[ResidualEdit, ...],
    ) -> ForwardCapture:
        return self._forward_capture(
            input_ids,
            tokenized,
            capture_position,
            edits=(),
            residual_edits=edits,
        )

    def _group_edits(
        self, edits: tuple[ActivationEdit, ...]
    ) -> dict[int, tuple[ActivationEdit, ...]]:
        grouped: dict[int, list[ActivationEdit]] = {}
        occupied: set[tuple[int, int]] = set()
        for edit in edits:
            if not 0 <= edit.layer < len(self.layers):
                raise ValueError(f"intervention layer {edit.layer} is out of range")
            width = int(self.layers[edit.layer].mlp.down_proj.in_features)
            if not edit.neurons:
                raise ValueError("an activation edit must select at least one neuron")
            for neuron in edit.neurons:
                if not 0 <= neuron < width:
                    raise ValueError(
                        f"intervention neuron L{edit.layer}:n{neuron} is out of range"
                    )
                identity = (edit.layer, neuron)
                if identity in occupied:
                    raise ValueError(f"duplicate activation edit for L{edit.layer}:n{neuron}")
                occupied.add(identity)
            if edit.operation == "mix":
                if edit.source_values is None:
                    raise ValueError("mix edits require source_values")
                source_count = int(edit.source_values.numel())
                if source_count not in {len(edit.neurons), width}:
                    raise ValueError(
                        "mix source_values must contain selected values or a full layer"
                    )
            grouped.setdefault(edit.layer, []).append(edit)
        return {layer: tuple(values) for layer, values in grouped.items()}

    @staticmethod
    def _apply_edits(
        tensor: torch.Tensor,
        *,
        position: int,
        edits: tuple[ActivationEdit, ...],
    ) -> torch.Tensor:
        if not edits:
            return tensor
        modified = tensor.clone()
        for edit in edits:
            indices = torch.tensor(
                edit.neurons,
                device=modified.device,
                dtype=torch.long,
            )
            current = modified[0, position, :].index_select(0, indices)
            if edit.operation == "scale":
                replacement = current * edit.strength
            else:
                assert edit.source_values is not None
                source = edit.source_values.detach().to(
                    device=modified.device,
                    dtype=modified.dtype,
                ).flatten()
                if source.numel() != len(edit.neurons):
                    source = source.index_select(0, indices)
                replacement = current * (1.0 - edit.strength) + source * edit.strength
            modified[0, position, :].index_copy_(0, indices, replacement)
        return modified

    def _group_residual_edits(
        self, edits: tuple[ResidualEdit, ...]
    ) -> dict[int, tuple[ResidualEdit, ...]]:
        grouped: dict[int, list[ResidualEdit]] = {}
        hidden_size = int(self.layers[0].mlp.down_proj.out_features)
        for edit in edits:
            if not 0 <= edit.layer < len(self.layers):
                raise ValueError(f"residual intervention layer {edit.layer} is out of range")
            if int(edit.direction.numel()) != hidden_size:
                raise ValueError(
                    f"residual direction has {edit.direction.numel()} values, expected {hidden_size}"
                )
            if not torch.isfinite(edit.direction).all().item():
                raise ValueError("residual direction contains non-finite values")
            if float(edit.direction.detach().float().norm().item()) <= 1e-12:
                raise ValueError("residual direction must have non-zero norm")
            grouped.setdefault(edit.layer, []).append(edit)
        return {layer: tuple(values) for layer, values in grouped.items()}

    @staticmethod
    def _apply_residual_edits(
        tensor: torch.Tensor,
        *,
        position: int,
        edits: tuple[ResidualEdit, ...],
    ) -> torch.Tensor:
        if not edits:
            return tensor
        modified = tensor.clone()
        for edit in edits:
            direction = edit.direction.detach().to(
                device=modified.device, dtype=modified.dtype
            ).flatten()
            if edit.normalization == "residual_norm":
                direction = direction / direction.float().norm().to(direction.dtype)
                amplitude = modified[0, position, :].float().norm().to(
                    modified.dtype
                ) * edit.beta
            else:
                amplitude = edit.beta
            modified[0, position, :] = (
                modified[0, position, :] + direction * amplitude
            )
        return modified

    @classmethod
    def _residual_output_hook(
        cls,
        edits: tuple[ResidualEdit, ...],
        *,
        position: int,
    ):
        def hook(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor | tuple[torch.Tensor, ...],
        ) -> torch.Tensor | tuple[torch.Tensor, ...]:
            tensor = output[0] if isinstance(output, tuple) else output
            modified = cls._apply_residual_edits(
                tensor, position=position, edits=edits
            )
            if isinstance(output, tuple):
                return (modified, *output[1:])
            return modified

        return hook

    def _forward_capture(
        self,
        input_ids: torch.Tensor,
        tokenized: TokenizedPrompt,
        capture_position: int,
        *,
        edits: tuple[ActivationEdit, ...],
        residual_edits: tuple[ResidualEdit, ...],
    ) -> ForwardCapture:
        sequence_length = int(input_ids.shape[-1])
        resolved_position = (
            capture_position
            if capture_position >= 0
            else sequence_length + capture_position
        )
        if not 0 <= resolved_position < sequence_length:
            raise ValueError(
                f"capture position {capture_position} is outside a prompt of "
                f"{sequence_length} tokens"
            )
        activations: dict[int, torch.Tensor] = {}
        last_layer_input: torch.Tensor | None = None
        last_ffn_output: torch.Tensor | None = None
        edits_by_layer = self._group_edits(edits)
        residual_edits_by_layer = self._group_residual_edits(residual_edits)

        def activation_hook(layer_index: int):
            def hook(
                _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
            ) -> tuple[torch.Tensor, ...] | None:
                tensor = self._apply_edits(
                    inputs[0],
                    position=resolved_position,
                    edits=edits_by_layer.get(layer_index, ()),
                )
                activations[layer_index] = (
                    tensor[0, resolved_position, :].detach().float().cpu()
                )
                if tensor is inputs[0]:
                    return None
                return (tensor, *inputs[1:])

            return hook

        def layer_input_hook(
            _module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
        ) -> None:
            nonlocal last_layer_input
            last_layer_input = inputs[0][0, resolved_position, :].detach().float().cpu()

        def ffn_output_hook(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            nonlocal last_ffn_output
            last_ffn_output = output[0, resolved_position, :].detach().float().cpu()

        with ExitStack() as stack:
            for index, layer in enumerate(self.layers):
                handle = layer.mlp.down_proj.register_forward_pre_hook(
                    activation_hook(index)
                )
                stack.callback(handle.remove)
            input_handle = self.layers[-1].register_forward_pre_hook(layer_input_hook)
            output_handle = self.layers[-1].mlp.down_proj.register_forward_hook(
                ffn_output_hook
            )
            stack.callback(input_handle.remove)
            stack.callback(output_handle.remove)
            for index, values in residual_edits_by_layer.items():
                residual_handle = self.layers[index].register_forward_hook(
                    self._residual_output_hook(values, position=resolved_position)
                )
                stack.callback(residual_handle.remove)

            with torch.inference_mode():
                outputs = self.model(
                    input_ids=input_ids.to(self.device),
                    use_cache=False,
                )

        if len(activations) != len(self.layers):
            missing = sorted(set(range(len(self.layers))) - set(activations))
            raise RuntimeError(f"failed to capture FFN activations for layers {missing}")
        if last_layer_input is None or last_ffn_output is None:
            raise RuntimeError("failed to capture last-layer FFN/Skip components")

        return ForwardCapture(
            logits=outputs.logits[0, -1, :].detach().float().cpu(),
            activations=tuple(activations[index] for index in range(len(self.layers))),
            last_layer_input=last_layer_input,
            last_ffn_output=last_ffn_output,
            tokenized=tokenized,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        seed: int,
        edits: tuple[ActivationEdit, ...] = (),
    ) -> GeneratedSequence:
        return self._generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            edits=edits,
            residual_edits=(),
        )

    def generate_residual_intervened(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        seed: int,
        edits: tuple[ResidualEdit, ...],
    ) -> GeneratedSequence:
        return self._generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            edits=(),
            residual_edits=edits,
        )

    def _generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        seed: int,
        edits: tuple[ActivationEdit, ...],
        residual_edits: tuple[ResidualEdit, ...],
    ) -> GeneratedSequence:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        edits_by_layer = self._group_edits(edits)
        residual_edits_by_layer = self._group_residual_edits(residual_edits)

        def edit_hook(layer_index: int):
            def hook(
                _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
            ) -> tuple[torch.Tensor, ...] | None:
                values = edits_by_layer.get(layer_index, ())
                if not values:
                    return None
                modified = self._apply_edits(
                    inputs[0], position=-1, edits=values
                )
                return (modified, *inputs[1:])

            return hook

        torch.manual_seed(seed)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "use_cache": True,
        }
        if do_sample:
            generation_kwargs.update({"temperature": temperature, "top_p": top_p})
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None and eos_token_id is not None:
            generation_kwargs["pad_token_id"] = eos_token_id

        with ExitStack() as stack:
            for index, layer in enumerate(self.layers):
                if index not in edits_by_layer:
                    continue
                handle = layer.mlp.down_proj.register_forward_pre_hook(edit_hook(index))
                stack.callback(handle.remove)
            for index, values in residual_edits_by_layer.items():
                residual_handle = self.layers[index].register_forward_hook(
                    self._residual_output_hook(values, position=-1)
                )
                stack.callback(residual_handle.remove)
            with torch.inference_mode():
                generated = self.model.generate(
                    input_ids=input_ids.to(self.device),
                    **generation_kwargs,
                )
        prompt_length = int(input_ids.shape[-1])
        new_ids = tuple(int(value) for value in generated[0, prompt_length:].tolist())
        text = self.tokenizer.decode(list(new_ids), skip_special_tokens=True)
        return GeneratedSequence(text=text, token_ids=new_ids)

    def behavioral_direction(self, observable: ResolvedObservable) -> torch.Tensor:
        output_embedding = self.model.get_output_embeddings()
        if output_embedding is None or not hasattr(output_embedding, "weight"):
            raise RuntimeError("model does not expose an output embedding weight")
        return behavioral_direction(
            output_embedding.weight,
            observable.target_ids,
            observable.control_ids,
        )

    def structural_couplings(
        self,
        direction: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        couplings: list[torch.Tensor] = []
        for layer in self.layers:
            weight = layer.mlp.down_proj.weight.detach()
            # Keep the forward pass in the selected model dtype, but accumulate the
            # structural projection in float32. This is particularly useful on MPS,
            # where the MVP loads model weights in float16 to control memory.
            local_direction = direction.to(device=weight.device, dtype=torch.float32)
            coupling = torch.matmul(local_direction, weight.float())
            couplings.append(coupling.detach().float().cpu())
        return tuple(couplings)

    def attention_metadata(self) -> AttentionMetadata:
        config = self.model.config
        try:
            output_heads = int(config.num_attention_heads)
            key_value_heads = int(config.num_key_value_heads)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                "Qwen config does not expose attention head counts"
            ) from exc
        if output_heads <= 0 or key_value_heads <= 0:
            raise ValueError("attention head counts must be positive")
        if output_heads % key_value_heads:
            raise ValueError("Qwen output heads must be divisible by key/value heads")
        first_attention = getattr(self.layers[0], "self_attn", None)
        if first_attention is None or not hasattr(first_attention, "o_proj"):
            raise ValueError("model layers do not expose self_attn.o_proj")
        projection_width = int(first_attention.o_proj.in_features)
        configured_dim = getattr(config, "head_dim", None)
        head_dim = (
            int(configured_dim)
            if configured_dim is not None
            else projection_width // output_heads
        )
        if projection_width != output_heads * head_dim:
            raise ValueError(
                "attention output projection width does not match head metadata"
            )
        return AttentionMetadata(
            layer_count=len(self.layers),
            output_head_count=output_heads,
            key_value_head_count=key_value_heads,
            head_dim=head_dim,
        )

    def attention_output_couplings(
        self, direction: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        metadata = self.attention_metadata()
        values: list[torch.Tensor] = []
        for index, layer in enumerate(self.layers):
            attention = getattr(layer, "self_attn", None)
            if attention is None or not hasattr(attention, "o_proj"):
                raise ValueError(f"layer {index} does not expose self_attn.o_proj")
            weight = attention.o_proj.weight.detach().float()
            if weight.shape[1] != metadata.output_head_count * metadata.head_dim:
                raise ValueError(
                    f"layer {index} attention output width does not match config"
                )
            local_direction = direction.detach().to(
                device=weight.device, dtype=torch.float32
            ).flatten()
            if local_direction.numel() != weight.shape[0]:
                raise ValueError(
                    f"behavioral direction width {local_direction.numel()} does not "
                    f"match layer {index} output width {weight.shape[0]}"
                )
            coupling = torch.matmul(local_direction, weight).reshape(
                metadata.output_head_count, metadata.head_dim
            )
            values.append(coupling.detach().float().cpu())
        return tuple(values)

    def _group_attention_edits(
        self,
        edits: tuple[AttentionHeadEdit, ...],
        *,
        sequence_length: int,
    ) -> dict[int, tuple[AttentionHeadEdit, ...]]:
        metadata = self.attention_metadata()
        grouped: dict[int, list[AttentionHeadEdit]] = {}
        occupied: set[tuple[int, int, int]] = set()
        for edit in edits:
            if not 0 <= edit.layer < len(self.layers):
                raise ValueError(f"attention edit layer {edit.layer} is out of range")
            if not edit.heads:
                raise ValueError("an attention edit must select at least one head")
            if not edit.positions:
                raise ValueError("an attention edit must select at least one position")
            resolved_positions = tuple(
                position if position >= 0 else sequence_length + position
                for position in edit.positions
            )
            if any(
                position < 0 or position >= sequence_length
                for position in resolved_positions
            ):
                raise ValueError("attention edit position is outside the prompt")
            if len(resolved_positions) != len(set(resolved_positions)):
                raise ValueError("attention edit positions must be unique")
            if len(edit.heads) != len(set(edit.heads)):
                raise ValueError("attention edit heads must be unique")
            for head in edit.heads:
                if not 0 <= head < metadata.output_head_count:
                    raise ValueError(
                        f"attention head L{edit.layer}:H{head} is out of range"
                    )
                for position in resolved_positions:
                    identity = (edit.layer, head, position)
                    if identity in occupied:
                        raise ValueError(
                            f"duplicate attention edit for L{edit.layer}:H{head} "
                            f"at position {position}"
                        )
                    occupied.add(identity)
            if edit.operation == "mix":
                if edit.source_values is None:
                    raise ValueError("attention mix edits require source_values")
                source = edit.source_values
                expected = (
                    len(resolved_positions),
                    len(edit.heads),
                    metadata.head_dim,
                )
                if source.ndim == 2 and len(resolved_positions) == 1:
                    source = source.unsqueeze(0)
                if tuple(source.shape) != expected:
                    raise ValueError(
                        "attention mix source_values must have shape "
                        f"{expected}, got {tuple(edit.source_values.shape)}"
                    )
            grouped.setdefault(edit.layer, []).append(
                AttentionHeadEdit(
                    layer=edit.layer,
                    heads=edit.heads,
                    operation=edit.operation,
                    strength=edit.strength,
                    positions=resolved_positions,
                    source_values=edit.source_values,
                )
            )
        return {layer: tuple(values) for layer, values in grouped.items()}

    @staticmethod
    def _apply_attention_edits(
        tensor: torch.Tensor,
        *,
        metadata: AttentionMetadata,
        edits: tuple[AttentionHeadEdit, ...],
    ) -> torch.Tensor:
        if not edits:
            return tensor
        shaped = tensor.reshape(
            tensor.shape[0],
            tensor.shape[1],
            metadata.output_head_count,
            metadata.head_dim,
        )
        modified = shaped.clone()
        for edit in edits:
            head_indices = torch.tensor(
                edit.heads, device=modified.device, dtype=torch.long
            )
            for source_index, position in enumerate(edit.positions):
                current = modified[0, position].index_select(0, head_indices)
                if edit.operation == "scale":
                    replacement = current * edit.strength
                else:
                    assert edit.source_values is not None
                    source = edit.source_values.detach()
                    if source.ndim == 2:
                        source = source.unsqueeze(0)
                    replacement_source = source[source_index].to(
                        device=modified.device, dtype=modified.dtype
                    )
                    replacement = (
                        current * (1.0 - edit.strength)
                        + replacement_source * edit.strength
                    )
                modified[0, position].index_copy_(
                    0, head_indices, replacement
                )
        return modified.reshape_as(tensor)

    def forward_attention_capture(
        self,
        input_ids: torch.Tensor,
        tokenized: TokenizedPrompt,
        capture_position: int,
        *,
        layers: tuple[int, ...] = (),
        edits: tuple[AttentionHeadEdit, ...] = (),
        include_attention_weights: bool = False,
    ) -> AttentionForwardCapture:
        metadata = self.attention_metadata()
        sequence_length = int(input_ids.shape[-1])
        resolved_position = (
            capture_position
            if capture_position >= 0
            else sequence_length + capture_position
        )
        if not 0 <= resolved_position < sequence_length:
            raise ValueError(
                f"capture position {capture_position} is outside a prompt of "
                f"{sequence_length} tokens"
            )
        requested_layers = set(layers or tuple(range(len(self.layers))))
        if any(layer < 0 or layer >= len(self.layers) for layer in requested_layers):
            raise ValueError("attention capture layer is out of range")
        edits_by_layer = self._group_attention_edits(
            edits, sequence_length=sequence_length
        )
        head_outputs: dict[int, torch.Tensor] = {}
        attention_weights: dict[int, torch.Tensor] = {}
        values: dict[int, torch.Tensor] = {}

        def output_projection_hook(layer_index: int):
            def hook(
                _module: torch.nn.Module,
                inputs: tuple[torch.Tensor, ...],
            ) -> tuple[torch.Tensor, ...] | None:
                tensor = self._apply_attention_edits(
                    inputs[0],
                    metadata=metadata,
                    edits=edits_by_layer.get(layer_index, ()),
                )
                if layer_index in requested_layers:
                    head_outputs[layer_index] = (
                        tensor[0]
                        .reshape(
                            sequence_length,
                            metadata.output_head_count,
                            metadata.head_dim,
                        )
                        .detach()
                        .float()
                        .cpu()
                    )
                if tensor is inputs[0]:
                    return None
                return (tensor, *inputs[1:])

            return hook

        def value_hook(layer_index: int):
            def hook(
                _module: torch.nn.Module,
                _inputs: tuple[torch.Tensor, ...],
                output: torch.Tensor,
            ) -> None:
                if layer_index not in requested_layers:
                    return
                values[layer_index] = (
                    output[0]
                    .reshape(
                        sequence_length,
                        metadata.key_value_head_count,
                        metadata.head_dim,
                    )
                    .detach()
                    .float()
                    .cpu()
                )

            return hook

        def attention_hook(layer_index: int):
            def hook(
                _module: torch.nn.Module,
                _inputs: tuple[torch.Tensor, ...],
                output: Any,
            ) -> None:
                if layer_index not in requested_layers:
                    return
                weights = (
                    output[1]
                    if isinstance(output, tuple) and len(output) > 1
                    else None
                )
                if weights is None or not torch.is_tensor(weights):
                    raise RuntimeError(
                        "eager attention did not return attention weights"
                    )
                attention_weights[layer_index] = (
                    weights[0, :, resolved_position, :]
                    .detach()
                    .float()
                    .cpu()
                )

            return hook

        previous_implementation = getattr(
            self.model.config, "_attn_implementation", None
        )
        with ExitStack() as stack:
            for index, layer in enumerate(self.layers):
                attention = getattr(layer, "self_attn", None)
                if attention is None or not hasattr(attention, "o_proj"):
                    raise ValueError(f"layer {index} does not expose self_attn.o_proj")
                if index in requested_layers or index in edits_by_layer:
                    handle = attention.o_proj.register_forward_pre_hook(
                        output_projection_hook(index)
                    )
                    stack.callback(handle.remove)
                if include_attention_weights and index in requested_layers:
                    if not hasattr(attention, "v_proj"):
                        raise ValueError(
                            f"layer {index} does not expose self_attn.v_proj"
                        )
                    value_handle = attention.v_proj.register_forward_hook(
                        value_hook(index)
                    )
                    attention_handle = attention.register_forward_hook(
                        attention_hook(index)
                    )
                    stack.callback(value_handle.remove)
                    stack.callback(attention_handle.remove)
            if include_attention_weights:
                self.model.config._attn_implementation = "eager"
            try:
                kwargs: dict[str, Any] = {
                    "input_ids": input_ids.to(self.device),
                    "use_cache": False,
                }
                with torch.inference_mode():
                    outputs = self.model(**kwargs)
            finally:
                if include_attention_weights:
                    self.model.config._attn_implementation = previous_implementation

        missing = requested_layers - set(head_outputs)
        if missing:
            raise RuntimeError(
                f"failed to capture attention head outputs for layers {sorted(missing)}"
            )
        if include_attention_weights:
            missing_weights = requested_layers - set(attention_weights)
            missing_values = requested_layers - set(values)
            if missing_weights or missing_values:
                raise RuntimeError(
                    "failed to capture eager attention tensors: "
                    f"weights={sorted(missing_weights)} values={sorted(missing_values)}"
                )

        return AttentionForwardCapture(
            logits=outputs.logits[0, -1, :].detach().float().cpu(),
            head_outputs=tuple(
                head_outputs.get(index) for index in range(len(self.layers))
            ),
            attention_weights=tuple(
                attention_weights.get(index) for index in range(len(self.layers))
            ),
            values=tuple(values.get(index) for index in range(len(self.layers))),
            tokenized=tokenized,
        )
