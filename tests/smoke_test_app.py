"""Headless test of app.py against the assignment's Streamlit requirements.

Uses Streamlit's AppTest (no browser needed) and checks that the app:
  * runs without errors for every model, including the misclassified-rows filter;
  * shows metrics from the SAVED models that match model/metrics.csv, both on the
    built-in test set and on uploaded CSVs (as-is, Kaggle column names, 1/0 labels);
  * rejects uploads that lack a feature column or the target column, or that
    have unknown labels, and reports dropped rows with bad values;
  * never trains a model: Pipeline.fit and GridSearchCV.fit raise for the whole run.

Run from the project root (after training):

    python tests/smoke_test_app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so app.py can import the `model` package

import pandas as pd  # noqa: E402
from sklearn.model_selection import GridSearchCV  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from model.common import METRIC_NAMES, METRICS_PATH, MODEL_FILES, TARGET, TEST_DATA_PATH  # noqa: E402


def _refuse_fit(self, *args, **kwargs):
    raise AssertionError(f"the app tried to train a model ({type(self).__name__}.fit)")


Pipeline.fit = _refuse_fit
GridSearchCV.fit = _refuse_fit

EXPECTED = pd.read_csv(METRICS_PATH).set_index("Model")
TEST = pd.read_csv(TEST_DATA_PATH)
failures: list[str] = []


def report(ok: bool, context: str, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'}  {context}")
    if not ok:
        if detail:
            print("      " + detail.replace("\n", "\n      "))
        failures.append(context)


def problems(at: AppTest) -> str:
    out = []
    for exc in at.exception:
        trace = "\n".join(getattr(exc, "stack_trace", None) or [])
        out.append(f"exception: {exc.value}\n{trace}")
    out += [f"st.error: {err.value}" for err in at.error]
    return "\n".join(out)


def upload(at: AppTest, frame: pd.DataFrame, name: str) -> AppTest:
    at.file_uploader[0].set_value((name, frame.to_csv(index=False).encode(), "text/csv"))
    return at.run()


def check_runs(at: AppTest, context: str) -> bool:
    errs = problems(at)
    report(not errs, context, errs)
    return not errs


def check_metrics(at: AppTest, context: str) -> None:
    """Every model's six metric tiles must equal model/metrics.csv (stored to 4 dp)."""
    for name in MODEL_FILES:
        at.selectbox[0].set_value(name).run()
        if not check_runs(at, f"{context} | {name} runs"):
            continue
        shown = {}
        for tile in at.metric:
            if tile.label in METRIC_NAMES:
                try:
                    shown[tile.label] = float(tile.value)
                except ValueError:
                    shown[tile.label] = tile.value
        wrong = {
            m: (shown.get(m), float(EXPECTED.loc[name, m]))
            for m in METRIC_NAMES
            if not isinstance(shown.get(m), float) or abs(shown[m] - EXPECTED.loc[name, m]) > 1.0001e-4
        }
        report(not wrong, f"{context} | {name} metrics match model/metrics.csv",
               f"(shown, expected): {wrong}")


def main() -> int:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300).run()
    check_runs(at, "initial load")
    check_metrics(at, "built-in test set")

    at.checkbox[0].check().run()
    check_runs(at, "show only misclassified rows")

    good_uploads = {
        "uploaded test_data.csv": TEST,
        "uploaded Kaggle-format CSV": TEST.rename(
            columns=lambda c: c.replace("concave_points", "concave points")
        ).assign(**{"Unnamed: 32": None}),
        "uploaded CSV with 1/0 labels": TEST.assign(**{TARGET: (TEST[TARGET] == "M").astype(int)}),
    }
    for context, frame in good_uploads.items():
        upload(at, frame, "upload.csv")
        uses_upload = any("uploaded file" in c.value for c in at.caption)
        report(uses_upload, f"{context} | app evaluates the uploaded file, not the default")
        check_metrics(at, context)

    bad_value = TEST.astype({"radius_mean": object})
    bad_value.loc[0, "radius_mean"] = "not a number"
    upload(at, bad_value, "bad_value.csv")
    warned = any("Dropped 1 row" in w.value for w in at.warning)
    report(warned and not problems(at), "upload with one non-numeric value | row dropped and reported",
           problems(at) or f"warnings: {[w.value for w in at.warning]}")

    bad_uploads = {
        "missing feature column": (TEST.drop(columns=["radius_mean"]), "missing 1 required feature"),
        "missing target column": (TEST.drop(columns=[TARGET]), f"no '{TARGET}' target column"),
        "unknown labels": (TEST.assign(**{TARGET: "X"}), "Unrecognised"),
    }
    for context, (frame, expected) in bad_uploads.items():
        upload(at, frame, "bad.csv")
        errors = [e.value for e in at.error]
        ok = not at.exception and any(expected in e for e in errors)
        report(ok, f"rejects upload with {context}",
               f"errors: {errors}, exceptions: {[e.value for e in at.exception]}")

    print("\nNo model was trained during the test: Pipeline.fit and GridSearchCV.fit were disabled.")
    print("PASSED" if not failures else f"FAILED ({len(failures)} check(s))")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
