from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_app_starts_and_runs_single_prediction():
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")
    demo_model = PROJECT_ROOT / "models/demo_baseline_rf.joblib"
    assert demo_model.exists(), "Run scripts/create_demo_assets.py before the app smoke test."

    app = streamlit_testing.AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30)
    app.run()
    assert not app.exception
    analyze_buttons = [button for button in app.button if button.label == "Analyze sequence"]
    assert len(analyze_buttons) == 1
    analyze_buttons[0].click().run()
    assert not app.exception
    rendered_markdown = "\n".join(item.value for item in app.markdown)
    assert "AMP Finder AI · Research Edition" in rendered_markdown
    assert "Toy mode is active" in rendered_markdown
    assert "Data Observatory" in rendered_markdown
    assert "Scholar Library" in rendered_markdown
    assert "Sequence-only limitation" in " ".join(
        dataframe.value.to_string() for dataframe in app.dataframe
    )
    assert app.get("plotly_chart")
