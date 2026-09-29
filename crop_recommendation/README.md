# AI-Driven Crop Recommendation System Using Soil and Climate Data

**Random Forest-Based Explainable Decision Support System for Crop Selection**

## Challenge
Challenge 42: Crop Recommendation from Soil and Climate.

## Problem Statement
Develop a data-driven system using soil nutrients, pH, rainfall, temperature and humidity to
generate reliable predictions, actionable intelligence and an explainable decision-support output.

## Objective
Input: soil and climate parameters. Output: **Recommended Crop Class**.

## Dataset
`data/dataset.csv` (supplied file `crop_recommendation_test_dataset__1_.csv`)

- 1,005 rows and 8 columns originally; 1,000 rows after cleaning
- 10 crop classes: banana, chickpea, coffee, cotton, grapes, kidneybeans, maize, mango, rice, watermelon
- Balanced: about 100 records per crop
- Full inspection output is in `outputs/dataset_summary.txt`

## Features
| Feature | Type | Description |
|---|---|---|
| N, P, K | input | Soil nutrients (nitrogen, phosphorus, potassium) |
| temperature | input | Temperature |
| humidity | input | Relative humidity |
| ph | input | Soil pH |
| rainfall | input | Rainfall |
| NPK_Total | engineered | N + P + K |
| NPK_Average | engineered | (N + P + K) / 3 |

## Target Variable
`label` = recommended crop (text, encoded with `LabelEncoder`).

## Technologies Used
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Joblib, Streamlit.
No deep learning, no APIs, no database.

## Round 1 - Data Understanding & EDA (`train_model.py`)
- **Inspection:** shape, columns, dtypes, head/tail, missing values, duplicates, classes.
- **Cleaning:** 3 missing values (1 in N, 1 in pH, 1 in rainfall) filled with the median; 5 exact duplicate rows removed;
  data types checked (all numeric columns are numeric); invalid values checked (negatives, pH outside 0-14,
  humidity outside 0-100, temperature outside -10 to 60): none found; label whitespace/case normalised.
- **Outliers (IQR method):** Q1, Q3, IQR, lower and upper bounds calculated for every feature.
  Outliers exist in P (10.6%), K (10.0%), pH (0.5%) and rainfall (0.1%). Almost all P and K outliers belong to
  the **grapes** crop, which naturally has very high P and K. They are real crop-specific values, so
  **no outliers were removed or clipped**.
- **Encoding:** there are no categorical input features. The target is encoded with `LabelEncoder`.
- **Statistics:** `describe()` plus median and per-crop means (in `dataset_summary.txt`).
- **EDA plots** (in `outputs/`): class distribution, correlation heatmap; (in `outputs/eda_plots/`):
  histograms, boxplots and one feature-vs-crop boxplot for each of the 7 features.
- **Feature selection:** all 7 soil/climate features are kept (they are the factors named in the challenge and their
  crop-wise means clearly differ). No ID column exists and the target is not used as an input, so there is no data leakage.

## Round 2 - Random Forest Model
- **Feature engineering** (`features.py`): `NPK_Total = N + P + K`, `NPK_Average = NPK_Total / 3`.
- **Pipeline:** feature engineering -> `StandardScaler` -> `RandomForestClassifier`.
  (Random Forest does not need scaling; the scaler is included so preprocessing is part of the saved pipeline and does not change predictions.)
- **Split:** `train_test_split(test_size=0.20, random_state=42, stratify=y)` -> 800 train / 200 test rows.
- **Random Forest:** `RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)`.
- **Saved model:** `models/crop_recommendation_pipeline.pkl` (pipeline + label encoder + feature information).

## Round 3 - Streamlit Intelligence Layer (`app.py`)
Sidebar inputs -> **Recommend Crop** button -> recommended crop, model prediction probability, top 3 crops,
input summary, input vs typical values of the predicted crop, important factors, text explanation,
model performance, confusion matrix, feature importance and EDA charts.

## Model Evaluation
Calculated on the 20% test set (200 records). Weighted averaging is used for precision, recall and F1.
The exact values are saved in `outputs/metrics.json` after training (run 1 gave 100% on all four metrics,
with 0 wrong predictions, and 99.90% 5-fold cross-validation accuracy).
The dataset's crops are very well separated in feature space (for example grapes have very high P and K, chickpea very low humidity),
which is why the accuracy is so high. Real farm data is usually noisier.

## Explainability
- Random Forest `feature_importances_` (chart in `outputs/feature_importance.png`). Top 5 in the trained model:
  P, K, NPK_Average, NPK_Total, humidity.
- The app compares the user's input with the typical range (10th-90th percentile) of the predicted crop in the training data.
- The text explanation is generated from the model results (no LLM).
- Importance shows what the model relied on; it does not prove causation.

## Project Structure
```
crop_recommendation/
├── data/dataset.csv
├── models/crop_recommendation_pipeline.pkl
├── outputs/
│   ├── dataset_summary.txt, metrics.json
│   ├── confusion_matrix.png, feature_importance.png
│   ├── class_distribution.png, correlation_heatmap.png
│   └── eda_plots/ (histograms, boxplots, feature-vs-crop plots)
├── features.py        (feature engineering shared by training and app)
├── train_model.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation
```bash
pip install -r requirements.txt
```

## Training
```bash
python train_model.py
```

## Running Application
```bash
streamlit run app.py
```

## Expected Output
- Training prints the dataset inspection, cleaning summary, IQR table, metrics, error analysis and feature importance,
  then creates the plots, `metrics.json`, `dataset_summary.txt` and the model file.
- The last training lines show an "Example Unseen Test Input" prediction after reloading the saved model.
- The app opens at http://localhost:8501.

## Jury Demonstration Steps
1. Run `python train_model.py` (or show that the outputs already exist) and briefly show `outputs/dataset_summary.txt`.
2. Run `streamlit run app.py`.
3. Show the **Model Performance** and **Visualizations** sections (confusion matrix, feature importance).
4. In the sidebar, ask the jury for **new values**, or use the example: N=90, P=40, K=40, temperature=25, humidity=80, pH=6.5, rainfall=200.
5. Click **🌾 Recommend Crop**. Show the recommended crop, the model prediction probability, the important factors, the input summary and the explanation.
   (For this example the model predicts rice with about 94.5% probability; the value can change slightly if you retrain.)
6. Change the values (for example high P and K) and click again to show that the prediction changes.
7. Explain: nothing is edited by hand - the input goes through the same saved pipeline used in training.
8. If a value is outside the training range, the app shows a warning that the prediction may be less reliable.

## Project Workflow
Dataset -> Cleaning -> EDA -> Feature Engineering -> Random Forest -> Evaluation -> Saved Model -> Streamlit -> New Input -> Crop Recommendation

## Challenge Requirement Check

| Round | Requirement | Status | Where |
|---|---|---|---|
| 1 | Data understanding (input/output defined) | Done | `train_model.py`, README |
| 1 | Dataset inspection (shape, dtypes, head/tail, classes) | Done | `outputs/dataset_summary.txt` |
| 1 | Data quality (missing, duplicates, dtypes, inconsistent values) | Done | Cleaning section |
| 1 | Outlier handling (IQR, documented decision) | Done | IQR section, boxplots |
| 1 | Encoding (target LabelEncoder; no categorical inputs) | Done | Encoding section |
| 1 | Descriptive statistics | Done | `describe()` + median |
| 1 | EDA | Done | Statistics and plots |
| 1 | Visualization | Done | `outputs/`, `outputs/eda_plots/` |
| 1 | Feature selection | Done | Relevant features section |
| 2 | Feature engineering | Done | `features.py` |
| 2 | Feature selection | Done | 7 raw + 2 engineered |
| 2 | Scaling | Done | `StandardScaler` in pipeline |
| 2 | Train/test split | Done | 0.20, seed 42, stratified |
| 2 | Random Forest | Done | 200 trees |
| 2 | Training | Done | `pipeline.fit` |
| 2 | Prediction | Done | Test-set predictions |
| 2 | Accuracy | Done | `metrics.json` |
| 2 | Precision | Done | `metrics.json` (weighted) |
| 2 | Recall | Done | `metrics.json` (weighted) |
| 2 | F1 | Done | `metrics.json` (weighted) |
| 2 | Error analysis | Done | Actual vs predicted, confused pairs |
| 2 | Feature importance | Done | `feature_importance.png` |
| 2 | Model saving | Done | `models/crop_recommendation_pipeline.pkl` |
| 3 | Load model | Done | `app.py` (joblib) |
| 3 | New unseen input | Done | Sidebar + reload test in training |
| 3 | Prediction | Done | Recommend Crop button |
| 3 | Important factors | Done | Top 5 features |
| 3 | Charts | Done | Visualizations tabs |
| 3 | Streamlit | Done | `app.py` |
| 3 | Complete pipeline | Done | Feature engineering + scaler + RF saved together |
| 3 | Jury testing | Done | "Test New Soil & Climate Profile" |
| 3 | No manual modification | Done | Input goes straight to saved pipeline |
| 3 | Final recommendation | Done | Crop + probability + explanation |
| 3 | Demonstration | Done | Jury steps above |

## Possible Future Work
Compare with other algorithms, use real field data with more noise, add season/region information.
