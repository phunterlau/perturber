# Run the UI

Run these commands from the repository root.

## First-time setup

```bash
uv sync --extra dev --locked
uv run --locked probe model fetch Qwen/Qwen3-0.6B \
  --max-download-bytes 2000000000
```

The model is stored in the project-local `.hf-cache/` directory.

## WebUI (recommended)

```bash
uv run --locked probe server start
uv run --locked probe server status
```

Open <http://127.0.0.1:8765/>. The two modes share the same local daemon and
immutable run store:

- **Quick Probe** runs the original two-prompt FFN ranking workflow. A completed
  rank run can be promoted into a research case without rerunning it.
- **Research Cases** creates a durable, multi-pair case with discovery,
  validation, and held-out splits. Use **Define** to edit the research contract
  or canonical YAML, then use **Evidence** to review preflight and explicitly
  run one stage at a time.

The **FFN Circuit** view separates observational neuron ranking, controlled
interventions, and residual-direction controllability. The **Attention Path**
view contains the head landscape, matched-control dose response, token routes,
sender/receiver builder, tokenizer alignment preview, and controlled path
result. Only verified backend claims receive causal styling.

Use **Provenance** to inspect immutable run lineage, download a bounded research
packet, or copy a continuation prompt for another agent. Replay mutation,
comparison, stability, sensitivity, and bulk export remain CLI-first and their
commands are included in the packet.

Stop the background server when finished:

```bash
uv run --locked probe server stop
```

If port 8765 is occupied, add the same alternative port to `start` and the URL,
for example `probe server start --port 8766` and
<http://127.0.0.1:8766/>. Logs are written to `.probe/server.log`.

## TUI compatibility mode

```bash
uv run --locked probe tui --device mps
```

Use `F5` to analyze, `Ctrl+S` to export, and `Ctrl+Q` to quit. The command
preloads Qwen on the main thread to avoid the macOS/Textual
`bad values in fds_to_keep` failure.
