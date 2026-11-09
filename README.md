# Breast Cancer Diagnosis: Comparing Five ML Classifiers

An end-to-end classification project. Five machine-learning models are trained on the same public dataset, evaluated with six metrics, and demonstrated in an interactive Streamlit web app.

## 1. Problem statement

Given 30 numeric measurements of cell nuclei taken from a digitised image of a fine-needle aspirate (FNA) of a breast mass, predict whether the tumour is **malignant (M)** or **benign (B)**. This is a **binary classification** problem. The positive class is *malignant*, so recall measures how many cancers the model catches.

## 2. Dataset

| | |
|---|---|
| **Name** | Breast Cancer Wisconsin (Diagnostic) |
| **Source** | UCI Machine Learning Repository: <https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic> (also on Kaggle as *uciml/breast-cancer-wisconsin-data*) |
| **Instances** | 569 (requirement: at least 500) |
| **Features** | 30 numeric, real-valued (requirement: at least 12) |
| **Target** | `diagnosis`: M = malignant (212, 37.3%), B = benign (357, 62.7%) |
| **Missing values** | None |
| **File** | [data/breast_cancer.csv](data/breast_cancer.csv) (raw `wdbc.data` with a header row added) |

Ten properties are measured for each nucleus: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension. Each is summarised three ways across the nuclei in the image: the **mean**, the **standard error** (`_se`) and the **worst** value (mean of the three largest). That gives 30 features such as `radius_mean`, `radius_se` and `radius_worst`. The `id` column is an identifier and is not used for training.

**Train/test split.** 80/20 stratified split with `random_state=42`: 455 training rows and 114 test rows. The test rows are saved as [data/test_data.csv](data/test_data.csv) so they can be uploaded to the app.

## 3. Models and evaluation metrics

All five models are trained on the same training split. Each one's hyper-parameters are tuned with 5-fold stratified cross-validation (scored on F1) on the **training data only**. Each model is then evaluated once on the untouched test set.

| Model | Preprocessing | Hyper-parameters searched |
|---|---|---|
| Logistic Regression | StandardScaler | `C` ∈ {0.01, 0.1, 1, 10, 100} |
| Decision Tree | none (scale-invariant) | `criterion`, `max_depth` ∈ {3, 4, 5, 6, 8, None}, `min_samples_leaf` ∈ {1, 2, 4, 8} |
| K-Nearest Neighbors | StandardScaler | `n_neighbors` ∈ {3, …, 21}, `weights` ∈ {uniform, distance} |
| Naive Bayes (Gaussian) | none (scale-invariant) | `var_smoothing` ∈ {1e-11, …, 1e-5} |
| Support Vector Machine (RBF) | StandardScaler | `C` ∈ {0.1, 1, 10, 100}, `gamma` ∈ {scale, 0.1, 0.01, 0.001} |

Scaling matters here because raw feature ranges differ by about six orders of magnitude (`area_worst` reaches ~4,000, `fractal_dimension_se` is ~0.003). Distance- and margin-based models (KNN, SVM) and regularised Logistic Regression would otherwise be dominated by the large-valued features. Gaussian Naive Bayes models each feature separately, and decision-tree splits are threshold-based, so neither is affected by per-feature scaling.

### Comparison table (test set, n = 114)

<!-- METRICS:START -->
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9449 | 0.9722 | 0.8333 | 0.8974 | 0.8504 |
| K-Nearest Neighbors | 0.9386 | 0.9825 | 0.9730 | 0.8571 | 0.9114 | 0.8688 |
| Naive Bayes (Gaussian) | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | 0.9825 | 0.9960 | 1.0000 | 0.9524 | 0.9756 | 0.9626 |
<!-- METRICS:END -->

Metric definitions (positive class = malignant):

- **Accuracy**: share of all predictions that are correct.
- **AUC**: area under the ROC curve, computed from predicted probabilities. Threshold-independent.
- **Precision**: of the tumours predicted malignant, the share that are malignant.
- **Recall**: of the malignant tumours, the share that were detected.
- **F1**: harmonic mean of precision and recall.
- **MCC**: Matthews correlation coefficient. Uses all four confusion-matrix cells and is robust to the 63/37 class imbalance.

### Tuned hyper-parameters and test-set confusion matrices

The test set has 72 benign and 42 malignant tumours. FN = malignant tumour predicted benign (missed cancer). FP = benign tumour predicted malignant (false alarm).

| ML Model Name | Tuned hyper-parameters | CV F1 (train) | TN | FP | FN | TP | Errors |
|---|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | `C=1` | 0.9643 | 71 | 1 | 3 | 39 | 4 |
| Decision Tree | `criterion=entropy, max_depth=6, min_samples_leaf=1` | 0.9345 | 71 | 1 | 7 | 35 | 8 |
| K-Nearest Neighbors | `n_neighbors=3, weights=uniform` | 0.9668 | 71 | 1 | 6 | 36 | 7 |
| Naive Bayes (Gaussian) | `var_smoothing=1e-9` | 0.9208 | 72 | 0 | 7 | 35 | 7 |
| Support Vector Machine | `C=10, gamma=0.01` (RBF) | 0.9699 | 72 | 0 | 2 | 40 | 2 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Second best overall (accuracy 0.9649, MCC 0.9245). It shares the highest AUC (0.9960) with the SVM, so it ranks malignant cases above benign ones just as well. It made 4 errors: 3 missed malignancies and 1 false alarm. Tuning kept the default regularisation (`C=1`). A linear model doing this well shows that, once standardised, the two classes are almost linearly separable. |
| Decision Tree | The weakest model: lowest or tied-lowest on all six metrics (accuracy 0.9298, F1 0.8974, MCC 0.8504). It missed 7 of the 42 malignant tumours (recall 0.8333). Its AUC (0.9449) is well below the others because a deep tree with near-pure leaves (`max_depth=6`, `min_samples_leaf=1`) outputs probabilities close to 0 or 1, which rank cases only coarsely. Its axis-parallel splits can only approximate the class boundary in steps, and a single deep tree has high variance. |
| K-Nearest Neighbors | Accuracy 0.9386 and MCC 0.8688, with 6 missed malignancies and 1 false alarm. Cross-validation chose `k=3`, which had the second-best CV F1 (0.9668), but test F1 fell to 0.9114. That is the largest CV-to-test drop of any model, and typical of a small, high-variance neighbourhood. With k=3 the predicted probability can only be 0, ⅓, ⅔ or 1, which caps AUC (0.9825). Distances are also dominated by redundant features: radius, perimeter and area are almost perfectly correlated (r = 0.97–0.998), so tumour size is effectively counted several times. |
| Naive Bayes (Gaussian) | Perfect precision (1.0000, no false alarms) but recall of only 0.8333: 7 of 42 malignant tumours missed, tied for worst. Yet its AUC is high (0.9934), so it ranks cases well but its 0.5 decision threshold is badly placed. A lower threshold would catch more cancers. This matches its core assumption failing on this data: the size features are strongly correlated (r ≈ 0.97–0.998) but are treated as independent evidence, which distorts the probabilities. It also had the lowest CV F1 (0.9208). |
| Support Vector Machine | Best model on every metric: accuracy 0.9825, recall 0.9524, F1 0.9756, MCC 0.9626. It ties Logistic Regression on AUC (0.9960) and Naive Bayes on precision (1.0000). Only 2 errors on the test set, both missed malignancies, and no false alarms. It also had the best CV F1 (0.9699), so its lead holds across folds and isn't an artefact of one split. The tuned RBF kernel (`C=10`, `gamma=0.01`, about a third of scikit-learn's default gamma for 30 standardised features) gives a smooth maximum-margin boundary that suits these well-separated, scaled features. |

**Overall**

- **Ranking by MCC:** SVM (0.9626) > Logistic Regression (0.9245) > Naive Bayes (0.8715) > KNN (0.8688) > Decision Tree (0.8504). Accuracy gives the same order, with KNN and Naive Bayes tied at 0.9386.
- **Almost all errors are missed cancers.** Across the five models there are 25 false negatives against only 3 false positives. Precision is at least 0.97 for every model, so **recall (0.83–0.95) is what separates them**. For a diagnostic aid a missed malignancy costs far more than a false alarm, which makes the SVM the clear choice here. The high-AUC models (Naive Bayes in particular) could also trade a little precision for recall by lowering the decision threshold.
- **The test set is small**: 114 rows, 42 of them malignant. One missed malignancy moves recall by 2.4 points, and one error moves accuracy by 0.9 points. KNN (7 errors), Naive Bayes (7) and the Decision Tree (8) are within one misclassification of each other, so their order is not meaningful on this sample. The CV F1 column gives a second view. There, KNN ranks second and Naive Bayes last, while the SVM leads on both.

## 4. Streamlit app

`app.py` provides:

- **Model selection dropdown** for the five trained models.
- **CSV upload** for test data. The file needs the 30 feature columns. Include a `diagnosis` column (M/B or 1/0) to get metrics; without it, the app still shows predictions. The Kaggle copy of the dataset can be uploaded unchanged. A **sample test CSV** (the held-out split) can be downloaded from the sidebar. If nothing is uploaded, the app uses the held-out test set.
- **Model comparison tab**: all six metrics for all five models on the current data, with the best score per metric highlighted. Also a grouped bar chart and overlaid ROC curves.
- **Selected model tab**: tuned hyper-parameters, the six metrics, a confusion matrix and a classification report.
- **Predictions tab**: per-row prediction with the probability of malignancy, a filter for misclassified rows, and CSV download.
- **Dataset tab**: dataset description, class balance, feature descriptions and summary statistics.

The app loads the saved pipelines from `model/`. If they are missing, or were saved with a different scikit-learn version than the one installed, it retrains them in memory with the same settings and seed. This takes a few seconds, so a deployment never breaks on a pickle version mismatch.

## 5. Project structure

```
ML_ASSIGNMENT_2/
├── app.py                    # Streamlit web application
├── colab_training.ipynb      # runs training + app smoke test on Google Colab
├── requirements.txt
├── README.md
├── tests/
│   └── smoke_test_app.py     # headless Streamlit AppTest of app.py
├── data/
│   ├── breast_cancer.csv     # full dataset (569 × 32)
│   └── test_data.csv         # held-out test split, generated by training
└── model/
    ├── common.py             # feature list, label encoding, metric computation
    ├── train_models.py       # trains, tunes, evaluates and saves all five models
    ├── *.pkl                 # fitted pipelines, generated by training
    ├── metrics.csv           # test-set metrics, generated by training
    └── metadata.json         # split sizes, tuned hyper-parameters, versions
```

## 6. Training the models

**On Google Colab.** Open `colab_training.ipynb` in Colab, upload `colab_upload.zip` when prompted, and run every cell. The notebook trains the models, smoke-tests the app, pins `requirements.txt` to the scikit-learn version used, and downloads `trained_artifacts.zip`. Unzip that over the project folder.

**Locally.**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m model.train_models    # trains models, writes model/*.pkl, metrics, and updates this README's table
python tests/smoke_test_app.py  # optional: headless check that the app runs without errors
streamlit run app.py            # opens http://localhost:8501
```

## 7. Deploying to Streamlit Community Cloud

1. Push this folder to a public GitHub repository. Include `model/*.pkl`, `model/metadata.json` and `data/`.
2. Go to <https://share.streamlit.io>, choose **Create app** (or **New app**), then select the repository, the `main` branch and the main file `app.py`.
3. Under **Advanced settings**, choose the Python version the models were trained with. It's recorded in `model/metadata.json` under `versions.python`.
4. Deploy. Streamlit Cloud installs `requirements.txt` automatically.
