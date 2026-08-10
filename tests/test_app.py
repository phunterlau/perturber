import asyncio

from probing.app import ProbeWorkbench
from probing.samples import AGREEMENT_CAPITAL
from helpers import FakeAdapter, fake_result


def test_app_renders_result_tables(tmp_path) -> None:
    async def exercise() -> None:
        app = ProbeWorkbench(
            model_id="fake/qwen3",
            sample=AGREEMENT_CAPITAL,
            engine_factory=lambda: FakeAdapter(),  # not invoked in this test
            output_root=tmp_path,
            top_k=2,
        )
        async with app.run_test(size=(160, 50)) as pilot:
            app._show_result(fake_result())
            await pilot.pause()
            assert "Complete" in str(app.query_one("#status").render())
            assert app.query_one("#layer-table").row_count == 2
            assert app.query_one("#neuron-table").row_count == 2
            assert "Measured delta F" in str(app.query_one("#overview").render())

    asyncio.run(exercise())
