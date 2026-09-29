"""
Streamlit dashboard - AI-Driven Crop Recommendation System
Run:  streamlit run app.py

The app loads the saved pipeline (feature engineering + scaler + Random Forest),
so new inputs get exactly the same processing as the training data.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
MODEL_FILE = BASE / "models" / "crop_recommendation_pipeline.pkl"
METRICS_FILE = BASE / "outputs" / "metrics.json"
OUT = BASE / "outputs"

# Friendly names and units. Units are only shown for columns where they are standard;
# nutrient units are not given in the dataset, so none are shown for N, P and K.
NAMES = {"N": "Nitrogen (N)", "P": "Phosphorus (P)", "K": "Potassium (K)",
         "temperature": "Temperature", "humidity": "Humidity", "ph": "pH",
         "rainfall": "Rainfall"}
UNITS = {"temperature": "°C", "humidity": "%", "ph": "pH scale", "rainfall": "mm"}


def nice(col):
    return NAMES.get(col, col)


def label_with_unit(col):
    return f"{nice(col)} ({UNITS[col]})" if col in UNITS else nice(col)


st.set_page_config(page_title="Crop Recommendation System", page_icon="🌱", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_metrics():
    return json.loads(METRICS_FILE.read_text())


# --------------------------------------------------------------------- header
st.title("🌱 AI-Driven Crop Recommendation System")
st.subheader("Random Forest-Based Soil and Climate Decision Support")
st.caption("Soil & Climate Based Intelligent Crop Recommendation")

if not MODEL_FILE.exists() or not METRICS_FILE.exists():
    st.error("Trained model not found. Please run `python train_model.py` first.")
    st.stop()

bundle = load_model()
metrics = load_metrics()
pipeline = bundle["pipeline"]
label_encoder = bundle["label_encoder"]
features = bundle["raw_features"]
info = bundle["input_info"]
importance = pd.Series(bundle["feature_importance"]).sort_values(ascending=False)

# -------------------------------------------------------------------- inputs
st.sidebar.header("Test New Soil & Climate Profile")
st.sidebar.write("Enter any values, including completely new ones, then click the button.")

user_input = {}
for col in features:
    lim_min, lim_max = info[col]["limit_min"], info[col]["limit_max"]
    user_input[col] = st.sidebar.number_input(
        label_with_unit(col),
        min_value=lim_min, max_value=lim_max,
        value=round(info[col]["median"], 2),
        step=0.1 if col == "ph" else 1.0,
        help=f"Training data range: {info[col]['data_min']:.1f} to {info[col]['data_max']:.1f}",
    )

if st.sidebar.button("🌾 Recommend Crop", type="primary", width="stretch"):
    # 1. collect input -> 2. DataFrame -> 3. saved pipeline -> 4. prediction + probability
    input_df = pd.DataFrame([user_input])[features]
    probabilities = pipeline.predict_proba(input_df)[0]
    crop_names = label_encoder.inverse_transform(pipeline.classes_)
    st.session_state["result"] = {
        "input": dict(user_input),
        "probabilities": pd.Series(probabilities, index=crop_names).sort_values(ascending=False),
    }

# ------------------------------------------------------------------- result
st.header("Recommendation")
result = st.session_state.get("result")

if result is None:
    st.info("Enter soil and climate values in the sidebar and click **🌾 Recommend Crop**.")
else:
    probs = result["probabilities"]
    crop = probs.index[0]
    inputs = result["input"]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🌾 Recommended Crop")
        st.success(f"## 🌱 {crop.title()}")
    with col2:
        st.markdown("### Model Prediction Probability")
        st.metric(label="Probability given by the Random Forest", value=f"{probs.iloc[0] * 100:.1f}%")
        st.caption("This is the model's estimate from patterns in the training data, not a guarantee.")

    # warn when the input is outside what the model has seen
    outside = [nice(c) for c in features
               if not info[c]["data_min"] <= inputs[c] <= info[c]["data_max"]]
    if outside:
        st.warning("These values are outside the range of the training data, so the prediction "
                   "may be less reliable: " + ", ".join(outside))

    st.markdown("#### Other likely crops")
    top3 = probs.head(3)
    st.table(pd.DataFrame({"Crop": top3.index.str.title(),
                           "Model prediction probability": [f"{p * 100:.1f}%" for p in top3.values]}))

    left, right = st.columns(2)
    with left:
        st.markdown("#### Input Summary")
        summary = pd.DataFrame({"Parameter": [nice(c) for c in features],
                                "Value": [f"{inputs[c]:g}" + (f" {UNITS[c]}" if c in UNITS else "")
                                          for c in features]})
        st.table(summary.set_index("Parameter"))

    # compare the input with typical values of the predicted crop (from training data)
    profile = bundle["crop_profiles"][crop]
    rows, matching, not_matching = [], 0, []
    for c in features:
        p10, p90, mean = profile[c]["p10"], profile[c]["p90"], profile[c]["mean"]
        if inputs[c] < p10:
            status = "below typical range"
        elif inputs[c] > p90:
            status = "above typical range"
        else:
            status = "within typical range"
        if status == "within typical range":
            matching += 1
        else:
            not_matching.append(nice(c))
        rows.append({"Parameter": nice(c), "Your value": round(inputs[c], 2),
                     f"Typical {crop} (mean)": round(mean, 2),
                     "Typical range (10th-90th %)": f"{p10:.1f} to {p90:.1f}", "Status": status})
    with right:
        st.markdown(f"#### Your input vs typical values for {crop.title()}")
        st.dataframe(pd.DataFrame(rows).set_index("Parameter"), width="stretch")

    # important factors from the trained Random Forest
    st.markdown("#### Important Factors")
    top5 = importance.head(5)
    st.write("These features had higher importance in the trained Random Forest model.")
    fcol1, fcol2 = st.columns([1, 2])
    with fcol1:
        st.table(pd.DataFrame({"Rank": range(1, len(top5) + 1),
                               "Feature": [nice(f) for f in top5.index],
                               "Importance": [f"{v:.3f}" for v in top5.values]}).set_index("Rank"))
    with fcol2:
        st.bar_chart(pd.DataFrame({"Importance": top5.values},
                                  index=[nice(f) for f in top5.index]))
    st.caption("Feature importance shows what the model relied on. It does not prove causation. "
               "NPK_Total and NPK_Average are engineered from N, P and K.")

    # simple dynamic explanation (no LLM: text is built from the model results)
    st.markdown("#### Explanation")
    text = (f"The model predicts **{crop.title()}** for the given soil and climate conditions, with a "
            f"model prediction probability of {probs.iloc[0] * 100:.1f}%. The prediction is based on "
            f"patterns learned from the training dataset. The most influential features in the "
            f"trained model are {', '.join(nice(f) for f in top5.index[:3])}. "
            f"{matching} of your {len(features)} input values are within the typical range of "
            f"{crop.title()} in the training data")
    text += (f"; values outside the typical range: {', '.join(not_matching)}." if not_matching else ".")
    st.info(text)

# ------------------------------------------------------- model performance
st.header("Model Performance")
st.write(f"Calculated on the held-out test set ({metrics['test_rows']} records, 20% of the data) "
         f"during training.")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
m2.metric("Precision (weighted)", f"{metrics['precision'] * 100:.2f}%")
m3.metric("Recall (weighted)", f"{metrics['recall'] * 100:.2f}%")
m4.metric("F1-score (weighted)", f"{metrics['f1_score'] * 100:.2f}%")
st.caption(f"Extra check: 5-fold cross-validation accuracy = {metrics['cv_accuracy_mean'] * 100:.2f}%. "
           f"Incorrect test predictions: {metrics['incorrect_predictions']}.")

# ------------------------------------------------------------ visualizations
st.header("Visualizations")
charts = {
    "Feature Importance": OUT / "feature_importance.png",
    "Confusion Matrix": OUT / "confusion_matrix.png",
    "Class Distribution": OUT / "class_distribution.png",
    "Correlation Heatmap": OUT / "correlation_heatmap.png",
    "Feature Distributions": OUT / "eda_plots" / "histograms.png",
    "Boxplots (Outliers)": OUT / "eda_plots" / "boxplots.png",
}
for tab, (name, path) in zip(st.tabs(list(charts)), charts.items()):
    with tab:
        if path.exists():
            st.image(str(path), width="stretch")
        else:
            st.warning(f"{path.name} not found. Run `python train_model.py`.")

# -------------------------------------------------------------------- about
st.header("About this Project")
st.write(
    "This system recommends a crop from soil nutrients (N, P, K), pH, temperature, humidity and "
    "rainfall. Data is cleaned and explored, two simple features (NPK_Total and NPK_Average) are "
    "added, and a **Random Forest** classifier (200 trees) is trained. The whole processing chain "
    "is saved as one pipeline, so any new input is processed exactly like the training data. "
    "The result is a decision-support tool, not a guarantee of crop success."
)
