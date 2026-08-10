from __future__ import annotations

import argparse
from pathlib import Path

from .app import ProbeWorkbench
from .artifacts import export_result
from .domain import ProbeSpec
from .engine import ProbeEngine
from .samples import SAMPLES, get_sample


PROJECT_DIRECTORY = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pair-first perturbation probing workbench"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float32", "float16", "bfloat16"),
    )
    parser.add_argument("--sample", default="agreement-capital", choices=sorted(SAMPLES))
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--output", type=Path, default=PROJECT_DIRECTORY / "runs")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_DIRECTORY / ".hf-cache")
    parser.add_argument("--raw-prompt", action="store_true")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--headless", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    sample = get_sample(args.sample)
    engine: ProbeEngine | None = None

    def create_engine() -> ProbeEngine:
        nonlocal engine
        if engine is None:
            engine = ProbeEngine.from_pretrained(
                args.model,
                revision=args.revision,
                device=args.device,
                dtype=args.dtype,
                cache_dir=str(args.cache_dir),
                local_files_only=args.local_files_only,
            )
        return engine

    if args.headless:
        print(f"Loading {args.model}...")
        result = create_engine().analyze(
            ProbeSpec(
                model_id=args.model,
                revision=args.revision,
                pair=sample.pair,
                observable=sample.observable,
                chat_template=not args.raw_prompt,
                enable_thinking=args.thinking,
                top_k=args.top_k,
            )
        )
        destination = export_result(result, args.output)
        print(f"F(original):  {result.original_gap:+.6f}")
        print(f"F(perturbed): {result.perturbed_gap:+.6f}")
        print(f"Measured delta F: {result.measured_delta:+.6f}")
        print(f"Predicted sum I:  {result.predicted_delta:+.6f}")
        print(f"Exported: {destination}")
        return

    # Preserve the validated macOS guard: initialize PyTorch/MPS before Textual.
    print(f"Loading {args.model} before starting the TUI...")
    create_engine()
    ProbeWorkbench(
        model_id=args.model,
        sample=sample,
        engine_factory=create_engine,
        output_root=args.output,
        revision=args.revision,
        chat_template=not args.raw_prompt,
        enable_thinking=args.thinking,
        top_k=args.top_k,
    ).run()
