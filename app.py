"""Interactive comparison of five classifiers on the Breast Cancer Wisconsin data.

The app loads the models saved by ``model/train_models.py`` and only ever
predicts with them. It never trains or refits a model, on uploaded data or
otherwise.

Run from the project root:

    streamlit run app.py
"""

import json

import altair as alt
import joblib
import pandas as pd
import sklearn
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from model.common import (
    BASE_FEATURES,
    CLASS_NAMES,
    FEATURE_GROUPS,
    FEATURES,
    ID_COL,
    METADATA_PATH,
    METRIC_NAMES,
    MODEL_DIR,
    MODEL_FILES,
    TARGET,
    TEST_DATA_PATH,
    compute_metrics,
    load_dataset,
    positive_scores,
    prepare_test_data,
)

st.set_page_config(page_title="Breast Cancer Classifier Comparison", page_icon="🩺", layout="wide")

MODEL_NAMES = list(MODEL_FILES)

# Chart colours: reference categorical slots 1-5 (blue, orange, aqua, yellow,
# magenta) stepped per theme, a one-hue blue sequential ramp whose light end
# sits on the light surface and whose dark end sits on the dark surface, and
# neutral inks. "surface" matches Streamlit's own background so cell gaps read
# as empty space.
THEMES = {
    "light": {
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
        "sequential": ["#cde2fb", "#104281"],
        "ink_on_low": "#0b0b0b", "ink_on_high": "#ffffff",
        "muted": "#898781", "surface": "#ffffff",
        "highlight": "rgba(42, 120, 214, 0.18)",
    },
    "dark": {
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"],
        "sequential": ["#0d366b", "#86b6ef"],
        "ink_on_low": "#f0efec", "ink_on_high": "#0b0b0b",
        "muted": "#898781", "surface": "#0e1117",
        "highlight": "rgba(57, 135, 229, 0.28)",
    },
}

METRIC_HELP = {
    "Accuracy": "Share of all cases classified correctly.",
    "AUC": "Area under the ROC curve: how well the model ranks malignant above benign "
           "cases across every threshold (0.5 = chance, 1.0 = perfect).",
    "Precision": "Of the cases predicted malignant, the share that really are malignant.",
    "Recall": "Of the truly malignant cases, the share the model caught (sensitivity).",
    "F1": "Harmonic mean of precision and recall.",
    "MCC": "Matthews correlation coefficient: correlation between predictions and truth "
           "using all four confusion-matrix cells (-1 to 1, 0 = chance). Robust to class imbalance.",
}

FEATURE_DESCRIPTIONS = {
    "radius": "Mean distance from the centre to points on the perimeter",
    "texture": "Standard deviation of grey-scale values",
    "perimeter": "Perimeter of the nucleus",
    "area": "Area of the nucleus",
    "smoothness": "Local variation in radius lengths",
    "compactness": "perimeter² / area − 1.0",
    "concavity": "Severity of concave portions of the contour",
    "concave_points": "Number of concave portions of the contour",
    "symmetry": "Symmetry of the nucleus",
    "fractal_dimension": "“Coastline approximation” − 1",
}


# --------------------------------------------------------------------------- data


@st.cache_resource(show_spinner="Loading saved models…")
def load_saved_models():
    """Return (models, metadata, default test DataFrame) from the files saved by training."""
    metadata = json.loads(METADATA_PATH.read_text())
    models = {name: joblib.load(MODEL_DIR / fname) for name, fname in MODEL_FILES.items()}
    return models, metadata, pd.read_csv(TEST_DATA_PATH)


@st.cache_data
def get_full_dataset() -> pd.DataFrame:
    return load_dataset()


# ------------------------------------------------------------------------- charts


def theme_mode() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:  # older Streamlit without st.context.theme
        return "light"


def series_scale(theme: dict) -> alt.Scale:
    # Fixed domain, so a model keeps its colour whatever else is on screen.
    return alt.Scale(domain=MODEL_NAMES, range=theme["series"])


def metrics_bar_chart(metrics: pd.DataFrame, theme: dict) -> alt.Chart:
    long = metrics.melt(id_vars="Model", var_name="Metric", value_name="Score").dropna()
    low = min(0.0, float(long["Score"].min()))  # MCC can go negative on odd uploads
    return (
        alt.Chart(long)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Metric:N", sort=METRIC_NAMES, title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("Model:N", sort=MODEL_NAMES, scale=alt.Scale(paddingInner=0.12)),
            y=alt.Y("Score:Q", scale=alt.Scale(domain=[low, 1]), title="Score"),
            color=alt.Color("Model:N", scale=series_scale(theme),
                            legend=alt.Legend(orient="bottom", title=None)),
            tooltip=[alt.Tooltip("Score:Q", format=".4f"), "Model:N", "Metric:N"],
        )
        .properties(height=360)
    )


def roc_chart(y, scores: dict, theme: dict) -> alt.Chart:
    frames = []
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(y, s)
        frames.append(pd.DataFrame({"Model": name, "FPR": fpr, "TPR": tpr, "step": range(len(fpr))}))
    data = pd.concat(frames, ignore_index=True)

    axes = {
        "x": alt.X("FPR:Q", title="False positive rate (1 − specificity)", scale=alt.Scale(domain=[0, 1])),
        "y": alt.Y("TPR:Q", title="True positive rate (recall)", scale=alt.Scale(domain=[0, 1])),
    }
    chance = (
        alt.Chart(pd.DataFrame({"FPR": [0, 1], "TPR": [0, 1]}))
        .mark_line(color=theme["muted"], strokeWidth=1)
        .encode(**axes)
    )
    lines = (
        alt.Chart(data)
        .mark_line(strokeWidth=2)
        .encode(**axes, order="step:Q",
                color=alt.Color("Model:N", scale=series_scale(theme),
                                legend=alt.Legend(orient="bottom", title=None)))
    )
    # Invisible, generously sized points so hovering near a curve shows its values.
    hover = (
        alt.Chart(data)
        .mark_circle(size=250, opacity=0)
        .encode(**axes, tooltip=["Model:N", alt.Tooltip("FPR:Q", format=".3f"),
                                 alt.Tooltip("TPR:Q", format=".3f")])
    )
    return (chance + lines + hover).properties(height=380)


def confusion_chart(cm, theme: dict) -> alt.Chart:
    rows = []
    for i, actual in enumerate(CLASS_NAMES):
        total = cm[i].sum()
        for j, predicted in enumerate(CLASS_NAMES):
            rate = cm[i, j] / total if total else 0.0
            rows.append({
                "Actual": actual, "Predicted": predicted, "Count": int(cm[i, j]), "Rate": rate,
                "Ink": theme["ink_on_high"] if rate > 0.5 else theme["ink_on_low"],
            })
    data = pd.DataFrame(rows)

    base = alt.Chart(data).encode(
        x=alt.X("Predicted:N", sort=CLASS_NAMES, title="Predicted",
                axis=alt.Axis(orient="top", labelAngle=0)),
        y=alt.Y("Actual:N", sort=CLASS_NAMES, title="Actual"),
        tooltip=[alt.Tooltip("Count:Q"),
                 alt.Tooltip("Rate:Q", format=".1%", title="Share of actual class"),
                 "Actual:N", "Predicted:N"],
    )
    cells = base.mark_rect(cornerRadius=4, stroke=theme["surface"], strokeWidth=2).encode(
        color=alt.Color("Rate:Q", scale=alt.Scale(domain=[0, 1], range=theme["sequential"]), legend=None)
    )
    counts = base.mark_text(fontSize=22, fontWeight="bold", dy=-9).encode(
        text="Count:Q", color=alt.Color("Ink:N", scale=None)
    )
    rates = base.mark_text(fontSize=12, dy=13).encode(
        text=alt.Text("Rate:Q", format=".1%"), color=alt.Color("Ink:N", scale=None)
    )
    return (cells + counts + rates).properties(height=300)


# ---------------------------------------------------------------------------- app

st.title("🩺 Breast cancer diagnosis: comparing five classifiers")
st.markdown(
    "Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes and an RBF "
    "Support Vector Machine, trained on the **Breast Cancer Wisconsin (Diagnostic)** dataset to "
    "classify a tumour as **benign** or **malignant** from 30 cell-nucleus measurements."
)

try:
    models, metadata, default_test = load_saved_models()
except Exception as exc:
    st.error(f"Couldn't load the saved models ({exc}). "
             "Create them first with `python -m model.train_models`.")
    st.stop()
theme = THEMES[theme_mode()]

with st.sidebar:
    st.header("Controls")
    model_name = st.selectbox("Model", MODEL_NAMES,
                              help="Used in the Selected model and Predictions tabs.")
    uploaded = st.file_uploader(
        "Upload test data (CSV)", type="csv",
        help=f"Must contain the 30 feature columns and the '{TARGET}' target column "
             "(M/B or 1/0). The Kaggle copy of this dataset works as-is.",
    )
    st.download_button(
        "Download sample test CSV", data=default_test.to_csv(index=False).encode(),
        file_name="test_data.csv", mime="text/csv",
        help="The held-out 20% test split the models never saw during training.",
    )
    st.divider()
    st.caption(
        f"Predictions come from the **saved models** in `model/`. They were trained on "
        f"{metadata['n_train']} rows, with hyper-parameters tuned by {metadata['cv_folds']}-fold "
        f"stratified CV ({metadata['cv_scoring']}). The app never retrains them, including on "
        "uploaded data."
    )
    saved_version = metadata.get("sklearn_version")
    if saved_version != sklearn.__version__:
        st.warning(f"The models were saved with scikit-learn {saved_version}, but "
                   f"{sklearn.__version__} is installed. Install scikit-learn=={saved_version} "
                   "for reliable results.")

if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Couldn't read the CSV: {exc}")
        st.stop()
    source = f"uploaded file **{uploaded.name}**"
else:
    raw = default_test
    source = "the built-in **held-out test set**"

try:
    data, X, y, notes = prepare_test_data(raw)
except ValueError as exc:
    st.error(f"{exc} Download the sample test CSV from the sidebar to see the expected format.")
    st.stop()

st.caption(f"Evaluating the saved models on {source}: {len(X)} rows "
           f"({int(y.sum())} malignant, {int((y == 0).sum())} benign).")
for note in notes:
    st.warning(note)

# Predictions from every saved model on the active data (cheap: a few hundred rows at most).
preds = {name: m.predict(X) for name, m in models.items()}
scores = {name: positive_scores(m, X) for name, m in models.items()}
comparison = pd.DataFrame(
    [{"Model": n, **compute_metrics(y, preds[n], scores[n])} for n in MODEL_NAMES]
)
both_classes = y.nunique() == 2

tab_compare, tab_model, tab_preds, tab_data = st.tabs(
    ["📊 Model comparison", "🔍 Selected model", "🧾 Predictions", "📁 Dataset"]
)

with tab_compare:
    st.subheader("Evaluation metrics for all five models")
    styled = (
        comparison.set_index("Model").style
        .format("{:.4f}", na_rep="—")
        .highlight_max(axis=0, props=f"background-color: {theme['highlight']}")
    )
    st.dataframe(styled)
    st.caption("Highlighted cells mark the best score for each metric (ties are all highlighted).")
    if not both_classes:
        st.caption("AUC is undefined because the data contains only one class.")

    left, right = st.columns(2)
    with left:
        st.markdown("**All metrics by model**")
        st.altair_chart(metrics_bar_chart(comparison, theme))
    with right:
        st.markdown("**ROC curves**")
        if both_classes:
            st.altair_chart(roc_chart(y, scores, theme))
        else:
            st.info("ROC curves need both classes in the data.")

with tab_model:
    st.subheader(model_name)
    params = metadata.get("models", {}).get(model_name, {}).get("best_params", {})
    if params:
        st.caption("Tuned hyper-parameters: " + ", ".join(f"`{k} = {v}`" for k, v in params.items()))

    row = comparison.set_index("Model").loc[model_name]
    for col, metric in zip(st.columns(len(METRIC_NAMES)), METRIC_NAMES):
        value = row[metric]
        col.metric(metric, "—" if pd.isna(value) else f"{value:.4f}", help=METRIC_HELP[metric])

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y, preds[model_name], labels=[0, 1])
        st.altair_chart(confusion_chart(cm, theme))
        tn, fp, fn, tp = cm.ravel()
        st.caption(f"{tp} malignant caught, {fn} malignant missed, "
                   f"{fp} benign flagged as malignant, {tn} benign correctly cleared. "
                   "Percentages are of each actual class.")
    with right:
        st.markdown("**Classification report**")
        report = classification_report(y, preds[model_name], labels=[0, 1],
                                       target_names=CLASS_NAMES, output_dict=True,
                                       zero_division=0)
        report.pop("accuracy", None)  # a scalar, not a per-class row
        report_df = pd.DataFrame(report).T
        report_df["support"] = report_df["support"].astype(int)
        st.dataframe(
            report_df,
            column_config={c: st.column_config.NumberColumn(format="%.4f")
                           for c in ("precision", "recall", "f1-score")},
        )

with tab_preds:
    st.subheader(f"Predictions: {model_name}")
    out = pd.DataFrame(index=range(len(X)))
    if ID_COL in data.columns:
        out["ID"] = data[ID_COL].values
    out["Actual"] = [CLASS_NAMES[v] for v in y]
    out["Predicted"] = [CLASS_NAMES[v] for v in preds[model_name]]
    out["P(malignant)"] = scores[model_name]
    out["Correct"] = out["Actual"] == out["Predicted"]

    st.caption(f"{int((preds[model_name] == 1).sum())} of {len(out)} predicted malignant. "
               f"{int((~out['Correct']).sum())} misclassified.")
    shown = out[~out["Correct"]] if st.checkbox("Show only misclassified rows") else out
    st.dataframe(
        shown, hide_index=True,
        column_config={"P(malignant)": st.column_config.ProgressColumn(
            "P(malignant)", min_value=0.0, max_value=1.0, format="%.3f",
            help="The model's estimated probability that the tumour is malignant.")},
    )
    st.download_button("Download predictions (CSV)", data=out.to_csv(index=False).encode(),
                       file_name=f"predictions_{MODEL_FILES[model_name].removesuffix('.pkl')}.csv",
                       mime="text/csv")

with tab_data:
    full = get_full_dataset()
    counts = full[TARGET].value_counts()
    st.subheader("Breast Cancer Wisconsin (Diagnostic)")
    st.markdown(
        "Source: [UCI Machine Learning Repository]"
        "(https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic). "
        "Each row describes the cell nuclei in a digitised image of a fine-needle aspirate "
        "of a breast mass. The target is the diagnosis: **M** = malignant, **B** = benign."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Instances", f"{len(full)}")
    c2.metric("Features", f"{len(FEATURES)}")
    c3.metric("Benign", f"{counts.get('B', 0)} ({counts.get('B', 0) / len(full):.1%})")
    c4.metric("Malignant", f"{counts.get('M', 0)} ({counts.get('M', 0) / len(full):.1%})")

    st.markdown(
        f"**Features**: 10 measurements, each summarised three ways "
        f"({', '.join(FEATURE_GROUPS.values()).lower()}), giving columns like "
        "`radius_mean`, `radius_se` and `radius_worst`. No missing values. "
        f"Split: {metadata['n_train']} training / {metadata['n_test']} test rows, stratified."
    )
    st.dataframe(
        pd.DataFrame({"Measurement": BASE_FEATURES,
                      "Description": [FEATURE_DESCRIPTIONS[f] for f in BASE_FEATURES]}),
        hide_index=True,
    )
    with st.expander("Summary statistics (full dataset)"):
        st.dataframe(full[FEATURES].describe().T.round(4))
    with st.expander(f"Preview of the data being evaluated ({len(data)} rows)"):
        st.dataframe(data, hide_index=True)
