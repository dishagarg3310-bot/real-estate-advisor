# Real Estate Investment Advisor

Predicts whether a property is a good investment (classification) and its estimated price in 5 years (regression) for 250,000 Indian property listings. Includes MLflow experiment tracking with a model registry, hyperparameter tuning, and a Streamlit app.

**Live app:** https://real-estate-advisor-cya6l7bxpfx8cbshbrmaqr.streamlit.app/

![App](docs/app.png)

## App features

| Tab | What it does |
|-----|--------------|
| Predict | Property form, good-investment probability, model confidence, 5-year price, CAGR, gain, comparison with city median, what-if price charts, SHAP explanation of the top drivers |
| Compare | Two properties side by side with a winner and price chart |
| Top picks | Filter the market, the model scores a sample and ranks the top 10 by growth or probability |
| Market explorer | Filters (state, BHK, price, size), KPIs, price-per-sq-ft heatmap, age trend, today vs 5-year prices |
| EDA | The 20 EDA questions with chart and finding for each |
| Batch | Upload a CSV, get verdicts and 5-year prices, download the results (`docs/sample_batch.csv` is a template) |
| Model insights | Model comparison, confusion matrix, feature importance |

## Methodology

1. **Preprocessing** (`src/01_load_check.py`, `src/02_clean_data.py`): duplicate and missing-value checks, price per sq ft recalculated in rupees, amenities split into a count and 5 flags, floor consistency flag.
2. **Targets** (`src/03_create_targets.py`): `Good_Investment` and `Future_Price_5Y` are built from transparent domain rules, because the dataset does not contain them.
3. **EDA** (`src/04` to `src/07`): the 20 questions from the brief, charts and written findings in `reports/`.
4. **Features** (`src/features.py`): 27 inputs (10 categorical, 17 numeric), one-hot encoding and scaling, 80/20 stratified split (200,000 train / 50,000 test). Targets and the columns that define them (`Assumed_Growth_Rate`, `Investment_Score`, `City_Median_Price_per_SqFt`), plus `ID`, `Locality`, `Year_Built`, `Price_per_SqFt` and `Amenities`, are never inputs. The script asserts this.
5. **Modelling** (`src/train_models.py`): 3 classifiers and 3 regressors, all logged to MLflow.
6. **Evaluation and registry** (`src/evaluate_and_register.py`): confusion matrix, permutation feature importance, best models registered in the MLflow model registry.
7. **Tuning** (`src/tune_models.py`): XGBoost random search, see below.
8. **App** (`app.py`): Streamlit app described above.

## Run locally

Put `india_housing_prices.csv` in `data/`, then:

```
pip install -r requirements.txt
python src/01_load_check.py
python src/02_clean_data.py
python src/03_create_targets.py
python src/features.py
python src/train_models.py
python src/evaluate_and_register.py
streamlit run app.py
mlflow ui --backend-store-uri sqlite:///mlflow.db --workers 1
```

Optional tuning: `pip install -r requirements-dev.txt` then `python src/tune_models.py`.

The repository contains the compressed dataset (`data/model_data.csv.gz`) and the two deployed models (`models/best_*.joblib`), so `streamlit run app.py` works right after install.

## Results (test set, 50,000 rows)

| Task | Model | Key metrics |
|------|-------|-------------|
| Classification (`Good_Investment`) | Logistic regression | accuracy 0.861, F1 0.856, ROC AUC 0.950 |
| | Random forest | accuracy 0.996, F1 0.996, ROC AUC 0.9995 |
| | **Hist. gradient boosting (deployed)** | accuracy 0.999, F1 0.999, ROC AUC 1.000 |
| | XGBoost (tuned) | accuracy 0.9988, F1 0.9988, ROC AUC 1.000 |
| Regression (`Future_Price_5Y`, lakhs) | Ridge | RMSE 9.51, MAE 6.72, R2 0.998 |
| | Random forest (best RMSE, 190 MB, not deployed) | RMSE 0.36, MAE 0.22, R2 1.000 |
| | **Hist. gradient boosting (deployed)** | RMSE 1.21, MAE 0.97, R2 1.000 |
| | XGBoost (tuned) | RMSE 0.93, MAE 0.77, R2 1.000 |

Random forest has the lowest RMSE, but its model file is 190 MB, so the deployed app uses hist. gradient boosting.

Confusion matrix: `reports/confusion_matrix.png`. Feature importance: `reports/feature_importance_*.csv`.

## Hyperparameter tuning (XGBoost)

Random search, 12 trials per task, validated on a split inside the training set; the test set is used once at the end. All trials are nested MLflow runs in the experiment `real-estate-advisor-tuning`. The tuned classifier shows no gain over hist. gradient boosting (the targets are rule-derived, so the ceiling is already reached). The tuned regressor improves RMSE from 1.21 to 0.93. The deployed app keeps hist. gradient boosting because the gain is small.

## Use cases

- Investors compare a listing with its city average and see a 5-year price forecast.
- Buyers filter and rank the market by state, BHK, price and size.
- Real estate platforms can screen new listings in bulk with the batch upload.

## Limitation

The targets are derived from price, size, location and a few other columns by fixed rules, and price and size are also model inputs. The EDA shows the dataset behaves like synthetic data: no original feature explains price in a meaningful way. The near-perfect scores therefore show that the models recovered the target rules, not that they predict real market outcomes. SHAP explanations confirm this, since price and price per sq ft dominate. Treat the app as a demo, not financial advice.