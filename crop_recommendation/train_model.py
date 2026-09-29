"""
AI-Driven Crop Recommendation System Using Soil and Climate Data
Random Forest-Based Explainable Crop Recommendation

Run:  python train_model.py

Steps
  Round 1: load data -> inspect -> clean -> IQR outliers -> statistics -> EDA plots
  Round 2: feature engineering -> pipeline -> train/test split -> Random Forest
           -> evaluation -> error analysis -> feature importance -> save model
"""
import json
from collections import Counter
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")                      # draw plots without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, StandardScaler

from features import add_features

# ---------------------------------------------------------------- paths
BASE = Path(__file__).parent
DATA_FILE = BASE / "data" / "dataset.csv"
MODEL_FILE = BASE / "models" / "crop_recommendation_pipeline.pkl"
OUT = BASE / "outputs"
EDA = OUT / "eda_plots"
OUT.mkdir(exist_ok=True)
EDA.mkdir(exist_ok=True)
MODEL_FILE.parent.mkdir(exist_ok=True)

# Physically possible ranges, used only to catch clearly invalid values.
# N, P, K and rainfall cannot be negative (no upper limit is enforced).
VALID_RANGES = {"ph": (0, 14), "humidity": (0, 100), "temperature": (-10, 60)}

report = []                                # text saved to dataset_summary.txt


def show(text=""):
    """Print a line and also keep it for outputs/dataset_summary.txt."""
    print(text)
    report.append(str(text))


def section(title):
    show("\n" + "=" * 70)
    show(title)
    show("=" * 70)


# =================================================================== ROUND 1
section("ROUND 1 - 1. OBJECTIVE")
show("Input : soil nutrients, pH, temperature, humidity, rainfall")
show("Output: Recommended Crop Class (multi-class classification)")

# ---------------------------------------------------------- 2. import dataset
section("2. DATASET UNDERSTANDING")
raw = pd.read_csv(DATA_FILE)
show(f"Dataset file : {DATA_FILE.name}")
show(f"Shape        : {raw.shape}")
show(f"Rows         : {raw.shape[0]}")
show(f"Columns      : {raw.shape[1]}")
show(f"Column names : {list(raw.columns)}")
show("\nData types:\n" + raw.dtypes.astype(str).to_string())
show("\nFirst 5 rows:\n" + raw.head().to_string())
show("\nLast 5 rows:\n" + raw.tail().to_string())
show("\nMissing values per column:\n" + raw.isna().sum().to_string())
show(f"\nDuplicate rows : {raw.duplicated().sum()}")
show("\nUnique values per column:\n" + raw.nunique().to_string())

# ------------------------------------------- detect target / column types
# The target is the text (non-numeric) column. Everything numeric is an input.
text_cols = [c for c in raw.columns if raw[c].dtype == object or str(raw[c].dtype) in ("str", "string")]
if len(text_cols) == 1:
    TARGET = text_cols[0]
elif "label" in raw.columns:
    TARGET = "label"
else:
    raise SystemExit("Could not detect the target column automatically.")

numeric_cols = [c for c in raw.columns if c != TARGET and pd.api.types.is_numeric_dtype(raw[c])]
categorical_cols = [c for c in raw.columns if c not in numeric_cols and c != TARGET]

show(f"\nTarget column        : {TARGET}")
show(f"Numerical columns    : {numeric_cols}")
show(f"Categorical inputs   : {categorical_cols if categorical_cols else 'none (target is the only text column)'}")
show(f"Number of classes    : {raw[TARGET].nunique()}")
show(f"Crop classes         : {sorted(raw[TARGET].dropna().unique())}")
show("\nClass distribution:\n" + raw[TARGET].value_counts().to_string())
show("\nMin / Max of numerical features:\n" + raw[numeric_cols].agg(["min", "max"]).T.to_string())

# ----------------------------------------------------------- 4. cleaning
section("4. DATA CLEANING")
df = raw.copy()

# (a) incorrect data types: force numeric columns to numbers
nan_before_convert = int(df[numeric_cols].isna().sum().sum())
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
new_nan = int(df[numeric_cols].isna().sum().sum()) - nan_before_convert
show(f"[Data types] values that could not be converted to numbers: {new_nan}")

# (b) inconsistent values: whitespace, empty strings, letter case in the target
df[TARGET] = df[TARGET].astype(str).str.strip().str.lower()
df[TARGET] = df[TARGET].replace({"": np.nan, "nan": np.nan})
show(f"[Inconsistent] empty/missing target labels: {int(df[TARGET].isna().sum())}")
df = df.dropna(subset=[TARGET])
show(f"[Inconsistent] class names after cleaning: {sorted(df[TARGET].unique())}")

# (c) impossible values -> set to NaN so they are handled like missing values
invalid_total = 0
for col in numeric_cols:
    low, high = VALID_RANGES.get(col, (0, None))       # default rule: not negative
    bad = df[col] < low
    if high is not None:
        bad = bad | (df[col] > high)
    invalid_total += int(bad.sum())
    df.loc[bad, col] = np.nan
    show(f"[Inconsistent] {col:<12} valid range ({low}, {high if high is not None else 'no limit'}): "
         f"{int(bad.sum())} invalid values")

# (d) duplicates
n_dups = int(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
show(f"[Duplicates] exact duplicate rows found and removed: {n_dups}")

# (e) missing values: numeric columns -> median (robust to outliers)
missing_before = df[numeric_cols].isna().sum()
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())
missing_after = df[numeric_cols].isna().sum()
show("[Missing] before -> after filling with the median:")
show(pd.DataFrame({"before": missing_before, "after": missing_after}).to_string())
show(f"\nCleaning summary: {raw.shape[0]} rows -> {df.shape[0]} rows "
     f"({n_dups} duplicates removed, {int(missing_before.sum())} missing values filled, "
     f"{invalid_total} invalid values found)")

# ---------------------------------------------------------- 5. outliers (IQR)
section("5. OUTLIER DETECTION (IQR METHOD)")
show("Method: Q1 = 25th percentile, Q3 = 75th percentile, IQR = Q3 - Q1.")
show("A value is a possible outlier if it is below Q1 - 1.5*IQR or above Q3 + 1.5*IQR.\n")
iqr_rows = []
for col in numeric_cols:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    is_out = (df[col] < lower) | (df[col] > upper)
    iqr_rows.append({"feature": col, "Q1": q1, "Q3": q3, "IQR": iqr, "Lower": lower,
                     "Upper": upper, "outliers": int(is_out.sum()),
                     "outlier_%": round(100 * is_out.mean(), 2)})
iqr_table = pd.DataFrame(iqr_rows).set_index("feature").round(2)
show(iqr_table.to_string())

show("\nAre the outliers real agricultural observations? Which crops do they belong to?")
for col in numeric_cols:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    is_out = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
    if is_out.sum() > 0:
        owners = df.loc[is_out, TARGET].value_counts().head(3).to_dict()
        show(f"  {col:<12}: {int(is_out.sum())} outliers, mostly from {owners}")
show("\nDECISION: no outlier is removed or clipped. All values are inside physically possible")
show("ranges (see cleaning step) and the outliers are concentrated in specific crops")
show("(for example very high P and K are typical for one crop). They are real crop-specific")
show("conditions that help the model, so deleting them would remove useful information.")

# ------------------------------------------------- 7. descriptive statistics
section("7. DESCRIPTIVE STATISTICS")
desc = df[numeric_cols].describe().T
desc["median"] = df[numeric_cols].median()
show(desc[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]].round(2).to_string())
show("\nMean of each feature per crop:\n" + df.groupby(TARGET)[numeric_cols].mean().round(1).to_string())
show("\nSimple observations:")
show(f"  - Most variable feature (highest std): {desc['std'].idxmax()}")
show(f"  - Crop classes are balanced ({df[TARGET].value_counts().min()} to "
     f"{df[TARGET].value_counts().max()} rows each), so accuracy is a fair metric.")
show("  - Crops differ clearly in their average N, P, K, humidity and rainfall (table above).")

# --------------------------------------------------------------- 8/9. EDA plots
section("8-9. EDA VISUALIZATIONS")
sns.set_theme(style="whitegrid")
order = df[TARGET].value_counts().index

# class distribution
plt.figure(figsize=(9, 5))
ax = sns.countplot(data=df, x=TARGET, order=order, hue=TARGET, palette="viridis", legend=False)
ax.bar_label(ax.containers[0])
plt.title("Crop Class Distribution")
plt.xlabel("Crop")
plt.ylabel("Number of records")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(OUT / "class_distribution.png", dpi=120)
plt.close()

# correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap of Numerical Features")
plt.tight_layout()
plt.savefig(OUT / "correlation_heatmap.png", dpi=120)
plt.close()

# histograms and boxplots of all numerical features
n = len(numeric_cols)
rows = int(np.ceil(n / 3))
fig, axes = plt.subplots(rows, 3, figsize=(14, 4 * rows))
for ax, col in zip(axes.ravel(), numeric_cols):
    sns.histplot(df[col], kde=True, ax=ax, color="seagreen")
    ax.set_title(f"Distribution of {col}")
for ax in axes.ravel()[n:]:
    ax.axis("off")
plt.tight_layout()
plt.savefig(EDA / "histograms.png", dpi=110)
plt.close()

fig, axes = plt.subplots(rows, 3, figsize=(14, 4 * rows))
for ax, col in zip(axes.ravel(), numeric_cols):
    sns.boxplot(x=df[col], ax=ax, color="lightgreen")
    ax.set_title(f"Boxplot of {col}")
for ax in axes.ravel()[n:]:
    ax.axis("off")
plt.tight_layout()
plt.savefig(EDA / "boxplots.png", dpi=110)
plt.close()

# feature vs crop (one plot per feature)
for col in numeric_cols:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x=TARGET, y=col, order=order, hue=TARGET, palette="Set3", legend=False)
    plt.title(f"{col} for each crop")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(EDA / f"{col}_vs_crop.png", dpi=110)
    plt.close()
show(f"Saved: class_distribution.png, correlation_heatmap.png, and {2 + n} plots in eda_plots/")

# ------------------------------------------------- 6. encoding / 10. features
section("6 & 10. ENCODING AND RELEVANT FEATURES")
show(f"Categorical input features: {categorical_cols if categorical_cols else 'none -> no encoding needed for inputs'}")
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df[TARGET])
show("Target encoded with LabelEncoder: " +
     ", ".join(f"{c}={i}" for i, c in enumerate(label_encoder.classes_)))

X = df[numeric_cols]                       # target and ID-like columns are NOT included
show(f"\nInput features (7 raw): {numeric_cols}")
show("No ID column exists and the target is not used as an input, so there is no data leakage.")
show("All features are kept: N, P, K (nutrients), pH, temperature, humidity and rainfall are")
show("the exact soil/climate factors named in the challenge, and the crop means above differ")
show("clearly for each of them.")

# ================================================================== ROUND 2
section("ROUND 2 - FEATURE ENGINEERING")
X_engineered = add_features(X)
new_features = [c for c in X_engineered.columns if c not in numeric_cols]
final_features = list(X_engineered.columns)
show(f"Engineered features: {new_features}")
show("  NPK_Total   = N + P + K   (overall nutrient richness of the soil)")
show("  NPK_Average = NPK_Total / 3 (average nutrient level)")
show(f"Final model features ({len(final_features)}): {final_features}")

# ------------------------------------------------------ split / pipeline / RF
section("ROUND 2 - TRAIN/TEST SPLIT AND RANDOM FOREST")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
show(f"Train rows: {len(X_train)}   Test rows: {len(X_test)}   (test_size=0.20, random_state=42, stratified)")

RF_PARAMS = {"n_estimators": 200, "random_state": 42, "n_jobs": -1}
pipeline = Pipeline([
    ("feature_engineering", FunctionTransformer(add_features)),   # adds NPK features
    ("scaler", StandardScaler()),                                  # scales numerical features
    ("random_forest", RandomForestClassifier(**RF_PARAMS)),
])
show(f"Random Forest configuration: {RF_PARAMS}")
show("Note: StandardScaler is included so preprocessing is part of the saved pipeline.")
show("Random Forest itself does not need scaling, so it does not change the predictions.")

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
show("Training finished.")

# --------------------------------------------------------------- evaluation
section("EVALUATION (on the 20% unseen test set)")
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
    "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
    "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
}
for name, value in metrics.items():
    show(f"{name:<10}: {value:.4f}")

class_names = list(label_encoder.classes_)
cls_report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
show("\nClassification report:\n" + cls_report)

cv_scores = cross_val_score(pipeline, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42))
show(f"Extra check - 5-fold cross-validation accuracy: mean {cv_scores.mean():.4f} (std {cv_scores.std():.4f})")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix (Test Set)")
plt.xlabel("Predicted crop")
plt.ylabel("Actual crop")
plt.xticks(rotation=40, ha="right")
plt.tight_layout()
plt.savefig(OUT / "confusion_matrix.png", dpi=120)
plt.close()

# ----------------------------------------------------------- error analysis
section("ERROR ANALYSIS")
proba_test = pipeline.predict_proba(X_test)
wrong = np.where(y_pred != y_test)[0]
errors = X_test.iloc[wrong].copy()
errors.insert(0, "actual_crop", label_encoder.inverse_transform(y_test[wrong]))
errors.insert(1, "predicted_crop", label_encoder.inverse_transform(y_pred[wrong]))
errors.insert(2, "prediction_probability_%", (proba_test[wrong].max(axis=1) * 100).round(1))
show(f"Incorrect predictions: {len(wrong)} out of {len(y_test)}")
if len(wrong) > 0:
    show(errors.round(2).to_string())
    pairs = Counter(zip(errors["actual_crop"], errors["predicted_crop"]))
    show("\nMost confused pairs (actual -> predicted):")
    means = df.groupby(TARGET)[numeric_cols].mean().round(1)
    for (a, p), count in pairs.most_common(5):
        show(f"  {a} -> {p}: {count} time(s)")
        show("    mean values  " + a + ": " + means.loc[a].to_dict().__str__())
        show("    mean values  " + p + ": " + means.loc[p].to_dict().__str__())
    show("These crops have similar average values on several features, which is a possible")
    show("reason for the confusion (compare the mean values printed above).")
else:
    show("The model made no mistakes on this test set.")

# ------------------------------------------------------- feature importance
section("FEATURE IMPORTANCE (Random Forest feature_importances_)")
rf = pipeline.named_steps["random_forest"]
importance = pd.Series(rf.feature_importances_, index=final_features).sort_values(ascending=False)
show(importance.round(4).to_string())
show(f"\nTop 5 features: {list(importance.head(5).index)}")
show("Note: importance shows which features helped the trained model separate crops.")
show("It does not prove that a feature causes a crop to grow well. NPK_Total/NPK_Average")
show("share information with N, P and K, so their importance is split among them.")

plt.figure(figsize=(8, 5))
imp_sorted = importance.sort_values()
plt.barh(imp_sorted.index, imp_sorted.values, color="seagreen")
plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig(OUT / "feature_importance.png", dpi=120)
plt.close()

# ----------------------------------------------------------------- save
section("SAVING METRICS AND MODEL")
metrics_json = {
    **{k: round(float(v), 4) for k, v in metrics.items()},
    "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
    "test_rows": int(len(y_test)),
    "train_rows": int(len(y_train)),
    "incorrect_predictions": int(len(wrong)),
    "classes": class_names,
    "confusion_matrix": cm.tolist(),
    "feature_importance": {k: round(float(v), 4) for k, v in importance.items()},
    "classification_report": classification_report(y_test, y_pred, target_names=class_names,
                                                   output_dict=True, zero_division=0),
    "misclassified_examples": errors.round(2).to_dict(orient="records"),
}
(OUT / "metrics.json").write_text(json.dumps(metrics_json, indent=2))
show(f"Saved {OUT / 'metrics.json'}")

# Information the Streamlit app needs (input ranges, typical values of each crop)
input_info = {}
for col in numeric_cols:
    low, high = VALID_RANGES.get(col, (0, None))
    if high is None:                                   # no physical limit -> allow 2x dataset max
        high = float(df[col].max()) * 2
    input_info[col] = {"data_min": float(df[col].min()), "data_max": float(df[col].max()),
                       "median": float(df[col].median()),
                       "limit_min": float(low), "limit_max": float(high)}
crop_profiles = {}
for crop, group in df.groupby(TARGET):
    crop_profiles[crop] = {col: {"mean": float(group[col].mean()),
                                 "p10": float(group[col].quantile(0.10)),
                                 "p90": float(group[col].quantile(0.90))} for col in numeric_cols}

bundle = {
    "pipeline": pipeline,              # feature engineering + scaler + Random Forest
    "label_encoder": label_encoder,    # number -> crop name
    "raw_features": numeric_cols,      # inputs the user must provide
    "final_features": final_features,  # features seen by the Random Forest
    "engineered_features": new_features,
    "feature_importance": importance.to_dict(),
    "input_info": input_info,
    "crop_profiles": crop_profiles,
    "target_name": TARGET,
}
joblib.dump(bundle, MODEL_FILE)
show(f"Saved {MODEL_FILE}")

# --------------------------------------------- reload + example unseen test
section("RELOAD TEST WITH EXAMPLE UNSEEN TEST INPUT (technical test only)")
loaded = joblib.load(MODEL_FILE)
example = {"N": 90, "P": 40, "K": 40, "temperature": 25, "humidity": 80, "ph": 6.5, "rainfall": 200}
example_df = pd.DataFrame([example])[loaded["raw_features"]]
already_in_data = bool((df[numeric_cols].round(2).eq(example_df.iloc[0].round(2)).all(axis=1)).any())
probs = loaded["pipeline"].predict_proba(example_df)[0]
best = int(np.argmax(probs))
show("Example Unseen Test Input (not real-world ground truth): " + str(example))
show(f"Exact row already in dataset? {already_in_data}")
show(f"Predicted crop: {loaded['label_encoder'].classes_[best]}  "
     f"(model prediction probability {probs[best] * 100:.1f}%)")

(OUT / "dataset_summary.txt").write_text("\n".join(report), encoding="utf-8")
print("\nAll done. Saved outputs/dataset_summary.txt")
