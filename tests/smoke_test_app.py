"""Headless smoke test for app.py using Streamlit's AppTest (no browser needed).

Loads the app, switches through every model and toggles the misclassified
filter, failing if any run raises an exception or shows an st.error.

Run from the project root (after training):

    python tests/smoke_test_app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so app.py can import the `model` package

from streamlit.testing.v1 import AppTest  # noqa: E402

from model.common import MODEL_FILES  # noqa: E402


def check(at: AppTest, context: str) -> bool:
    problems = []
    for exc in at.exception:
        trace = "\n".join(getattr(exc, "stack_trace", None) or [])
        problems.append(f"exception: {exc.value}\n{trace}")
    problems += [f"st.error: {err.value}" for err in at.error]
    if problems:
        print(f"FAIL  {context}")
        for problem in problems:
            print("      " + problem.replace("\n", "\n      "))
        return False
    print(f"ok    {context}")
    return True


def main() -> int:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300).run()
    ok = check(at, "initial load")

    for name in MODEL_FILES:
        at.sidebar.selectbox[0].set_value(name).run()
        ok = check(at, f"model: {name}") and ok

    at.checkbox[0].check().run()
    ok = check(at, "show only misclassified rows") and ok

    for note in at.sidebar.info:
        print(f"note  {note.value}")
    print(f"\n{len(at.metric)} metric tiles, {len(at.dataframe)} tables rendered on the last run")
    print("PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
