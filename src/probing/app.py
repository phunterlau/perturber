from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import traceback

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    LoadingIndicator,
    RichLog,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)

from .artifacts import export_result
from .domain import ObservableSpec, ProbeResult, ProbeSpec, PromptPair
from .engine import ProbeEngine
from .observables import parse_token_csv
from .samples import Sample


class ProbeWorkbench(App[None]):
    TITLE = "Perturbation Probing Workbench"
    SUB_TITLE = "Exploratory pair (N=1)"
    CSS = """
    Screen {
        layout: vertical;
    }

    #status-row {
        height: 3;
        padding: 0 1;
        background: $panel;
    }

    #status {
        width: 1fr;
        content-align: left middle;
    }

    #loading {
        width: 8;
        display: none;
    }

    #prompt-row {
        height: 13;
    }

    .prompt-column {
        width: 1fr;
        border: round $primary;
        padding: 0 1;
    }

    .prompt-column Label {
        height: 1;
        text-style: bold;
    }

    TextArea {
        height: 1fr;
    }

    #observable-row {
        height: 5;
        padding: 0 1;
    }

    .observable-column {
        width: 1fr;
        padding-right: 1;
    }

    #button-column {
        width: 26;
        align: center middle;
    }

    #button-column Button {
        width: 22;
        margin-bottom: 1;
    }

    TabbedContent {
        height: 1fr;
    }

    #overview {
        padding: 1 2;
    }

    DataTable {
        height: 1fr;
    }

    #neuron-pane {
        layout: vertical;
    }

    #neuron-table {
        height: 1fr;
    }

    #inspector {
        height: 7;
        padding: 1 2;
        border-top: solid $primary;
    }
    """

    BINDINGS = [
        ("f5", "analyze", "Analyze"),
        ("ctrl+s", "export", "Export"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(
        self,
        *,
        model_id: str,
        sample: Sample,
        engine_factory: Callable[[], ProbeEngine],
        output_root: Path,
        revision: str | None = None,
        chat_template: bool = True,
        enable_thinking: bool = False,
        top_k: int = 500,
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.sample = sample
        self.engine_factory = engine_factory
        self.output_root = output_root
        self.revision = revision
        self.chat_template = chat_template
        self.enable_thinking = enable_thinking
        self.top_k = top_k
        self._engine: ProbeEngine | None = None
        self.result: ProbeResult | None = None
        self.selected_layer: int | None = None
        self.last_export: Path | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="status-row"):
            yield Static(
                f"Model: {self.model_id} | sample: {self.sample.name} | "
                "exploratory pair N=1",
                id="status",
            )
            yield LoadingIndicator(id="loading")
        with Horizontal(id="prompt-row"):
            with Vertical(classes="prompt-column"):
                yield Label("ORIGINAL / CONTROL")
                yield TextArea(self.sample.pair.original, id="original-prompt")
            with Vertical(classes="prompt-column"):
                yield Label("PERTURBED / TREATMENT")
                yield TextArea(self.sample.pair.perturbed, id="perturbed-prompt")
        with Horizontal(id="observable-row"):
            with Vertical(classes="observable-column"):
                yield Label("Target tokens (comma-separated; contributes + to F)")
                yield Input(
                    value=",".join(self.sample.observable.target_tokens),
                    id="target-tokens",
                )
            with Vertical(classes="observable-column"):
                yield Label("Control tokens (comma-separated; contributes - to F)")
                yield Input(
                    value=",".join(self.sample.observable.control_tokens),
                    id="control-tokens",
                )
            with Vertical(id="button-column"):
                yield Button("Analyze [F5]", id="analyze", variant="primary")
                yield Button("Export [Ctrl+S]", id="export")
        with TabbedContent(initial="overview-tab"):
            with TabPane("Overview", id="overview-tab"):
                yield Static(
                    "Press F5 to run two forward passes. No parameters are mutated.",
                    id="overview",
                )
            with TabPane("Layers", id="layers-tab"):
                yield DataTable(id="layer-table")
            with TabPane("Neurons", id="neurons-tab"):
                with Vertical(id="neuron-pane"):
                    yield DataTable(id="neuron-table")
                    yield Static("Select a neuron to inspect its score.", id="inspector")
            with TabPane("Run log", id="log-tab"):
                yield RichLog(id="run-log", wrap=True, highlight=True)
        yield Footer()

    def on_mount(self) -> None:
        layer_table = self.query_one("#layer-table", DataTable)
        layer_table.cursor_type = "row"
        neuron_table = self.query_one("#neuron-table", DataTable)
        neuron_table.cursor_type = "row"
        self.query_one("#run-log", RichLog).write(self.sample.description)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "analyze":
            self.action_analyze()
        elif event.button.id == "export":
            self.action_export()

    def _make_spec(self) -> ProbeSpec:
        target = parse_token_csv(self.query_one("#target-tokens", Input).value)
        control = parse_token_csv(self.query_one("#control-tokens", Input).value)
        original = self.query_one("#original-prompt", TextArea).text
        perturbed = self.query_one("#perturbed-prompt", TextArea).text
        return ProbeSpec(
            model_id=self.model_id,
            revision=self.revision,
            pair=PromptPair(original=original, perturbed=perturbed),
            observable=ObservableSpec(
                name="custom:tui",
                target_tokens=target,
                control_tokens=control,
            ),
            chat_template=self.chat_template,
            enable_thinking=self.enable_thinking,
            top_k=self.top_k,
        )

    def action_analyze(self) -> None:
        try:
            spec = self._make_spec()
        except Exception as exc:
            self._show_error(str(exc))
            return
        self._set_busy(True, "Loading model / running paired forwards...")
        self.query_one("#run-log", RichLog).write(
            f"Analyze requested: {len(spec.pair.original)} / "
            f"{len(spec.pair.perturbed)} prompt characters"
        )
        self._run_analysis(spec)

    @work(thread=True, exclusive=True, group="analysis")
    def _run_analysis(self, spec: ProbeSpec) -> None:
        try:
            if self._engine is None:
                self._engine = self.engine_factory()
            result = self._engine.analyze(spec)
        except Exception as exc:
            self.call_from_thread(self._show_error, str(exc), traceback.format_exc())
            return
        self.call_from_thread(self._show_result, result)

    def _set_busy(self, busy: bool, status: str) -> None:
        self.query_one("#status", Static).update(status)
        self.query_one("#loading", LoadingIndicator).styles.display = (
            "block" if busy else "none"
        )
        self.query_one("#analyze", Button).disabled = busy

    @staticmethod
    def _number(value: float | None) -> str:
        return "undefined" if value is None else f"{value:+.5f}"

    def _show_result(self, result: ProbeResult) -> None:
        self.result = result
        self.selected_layer = None
        self._set_busy(
            False,
            f"Complete in {result.elapsed_seconds:.2f}s | "
            f"{result.total_neuron_count:,} neurons ranked | top {len(result.neurons)} retained",
        )
        target = ", ".join(
            f"{item.text!r}->{item.token_id}:{item.decoded!r}"
            for item in result.observable.target
        )
        control = ", ".join(
            f"{item.text!r}->{item.token_id}:{item.decoded!r}"
            for item in result.observable.control
        )
        warnings = "\n".join(f"[yellow]Warning:[/] {item}" for item in result.warnings)
        self.query_one("#overview", Static).update(
            "\n".join(
                [
                    "[bold]Observable movement[/]",
                    f"F(original)  {result.original_gap:+.5f}",
                    f"F(perturbed) {result.perturbed_gap:+.5f}",
                    f"Measured delta F  {result.measured_delta:+.5f}",
                    f"Predicted sum I   {result.predicted_delta:+.5f}",
                    f"Next(original)  {result.original_prediction.decoded!r} "
                    f"p={result.original_prediction.probability:.3f}",
                    f"Next(perturbed) {result.perturbed_prediction.decoded!r} "
                    f"p={result.perturbed_prediction.probability:.3f}",
                    "",
                    "[bold]FFN/Skip diagnostic[/]",
                    f"Original:  {self._number(result.ffn_skip_original)}",
                    f"Perturbed: {self._number(result.ffn_skip_perturbed)}",
                    f"Mean:      {self._number(result.ffn_skip_mean)}",
                    f"Regime:    {result.circuit_regime}",
                    "",
                    f"Target:  {target}",
                    f"Control: {control}",
                    f"Tokens: original {len(result.original.input_ids)}, "
                    f"perturbed {len(result.perturbed.input_ids)}",
                    "",
                    warnings,
                ]
            )
        )
        self._populate_layers(result)
        self._populate_neurons(result)
        log = self.query_one("#run-log", RichLog)
        log.write(
            f"Completed: measured delta={result.measured_delta:+.5f}, "
            f"predicted={result.predicted_delta:+.5f}, "
            f"FFN/Skip mean={self._number(result.ffn_skip_mean)}"
        )
        for warning in result.warnings:
            log.write(f"WARNING: {warning}")

    def _populate_layers(self, result: ProbeResult) -> None:
        table = self.query_one("#layer-table", DataTable)
        table.clear(columns=True)
        table.add_columns(
            "layer",
            "sum I",
            "sum |I|",
            "+ mass",
            "- mass",
            "top10 share",
            "top neuron",
            "max |I|",
            "||delta a||",
        )
        table.add_row("ALL", "", "", "", "", "", "", "", "", key="all")
        for item in sorted(result.layers, key=lambda row: row.absolute_sum, reverse=True):
            table.add_row(
                str(item.layer),
                f"{item.signed_sum:+.4f}",
                f"{item.absolute_sum:.4f}",
                f"{item.positive_mass:.4f}",
                f"{item.negative_mass:.4f}",
                f"{item.top_10_share:.1%}",
                str(item.top_neuron),
                f"{item.maximum_absolute:.4f}",
                f"{item.activation_delta_norm:.4f}",
                key=str(item.layer),
            )

    def _populate_neurons(self, result: ProbeResult) -> None:
        table = self.query_one("#neuron-table", DataTable)
        table.clear(columns=True)
        table.add_columns(
            "rank", "layer", "neuron", "I", "c", "delta a", "a original", "a perturbed"
        )
        values = (
            result.neurons
            if self.selected_layer is None
            else tuple(item for item in result.neurons if item.layer == self.selected_layer)
        )
        for item in values:
            table.add_row(
                str(item.rank),
                str(item.layer),
                str(item.neuron),
                f"{item.importance:+.6f}",
                f"{item.coupling:+.6f}",
                f"{item.activation_delta:+.6f}",
                f"{item.original_activation:+.6f}",
                f"{item.perturbed_activation:+.6f}",
                key=f"{item.layer}:{item.neuron}",
            )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if self.result is None:
            return
        key = str(event.row_key.value)
        if event.data_table.id == "layer-table":
            self.selected_layer = None if key == "all" else int(key)
            self._populate_neurons(self.result)
            label = "all layers" if self.selected_layer is None else f"layer {self.selected_layer}"
            self.query_one("#inspector", Static).update(
                f"Neuron table filtered to {label}."
            )
        elif event.data_table.id == "neuron-table":
            layer_text, neuron_text = key.split(":", maxsplit=1)
            layer, neuron = int(layer_text), int(neuron_text)
            item = next(
                score
                for score in self.result.neurons
                if score.layer == layer and score.neuron == neuron
            )
            direction = "target-promoting coupling" if item.coupling > 0 else "control-promoting coupling"
            self.query_one("#inspector", Static).update(
                "\n".join(
                    [
                        f"[bold]L{item.layer}:n{item.neuron}[/] | global rank {item.rank} | layer rank {item.layer_rank}",
                        f"c={item.coupling:+.7f} ({direction})",
                        f"a: {item.original_activation:+.7f} -> {item.perturbed_activation:+.7f} "
                        f"(delta {item.activation_delta:+.7f})",
                        f"I = c * delta a = {item.importance:+.7f}",
                    ]
                )
            )

    def action_export(self) -> None:
        if self.result is None:
            self._show_error("Run an analysis before exporting.")
            return
        try:
            self.last_export = export_result(self.result, self.output_root)
        except Exception as exc:
            self._show_error(f"Export failed: {exc}")
            return
        self.query_one("#status", Static).update(f"Exported to {self.last_export}")
        self.query_one("#run-log", RichLog).write(f"Exported: {self.last_export}")

    def _show_error(self, message: str, details: str | None = None) -> None:
        self._set_busy(False, f"Error: {message}")
        self.query_one("#run-log", RichLog).write(f"ERROR: {message}")
        if details:
            self.query_one("#run-log", RichLog).write(details.rstrip())
        self.notify(message, title="Probe error", severity="error", timeout=8)
