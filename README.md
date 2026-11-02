\# Real Estate Investment Advisor



Predicts whether a property is a good investment (classification) and its estimated price in 5 years (regression). Includes MLflow experiment tracking with a model registry and a Streamlit app.



!\[App](docs/app.png)



\## Methodology



1\. \*\*Preprocessing:\*\* duplicates and missing values handled, derived columns built (price per sq ft, infrastructure score, amenity flags, floor consistency), and the two targets created from domain rules (`src/03\_create\_targets.py`).

2\. \*\*EDA:\*\* the 20 questions from the brief are answered in `reports/eda\_findings.md`.

3\. \*\*Features:\*\* 27 inputs (10 categorical, 17 numeric), one-hot encoding and scaling, 80/20 stratified split (`src/features.py`). Targets and the columns that define them are never inputs; the script asserts this.

4\. \*\*Modelling:\*\* 3 classifiers and 3 regressors, all logged to MLflow (`src/train\_models.py`). The best of each is registered in the MLflow model registry (`src/evaluate\_and\_register.py`).

5\. \*\*App:\*\* prediction form with confidence score, market explorer with filters, heatmap and trend charts, and a model insights tab with confusion matrix and feature importance (`app.py`).



\## Run



```

pip install -r requirements.txt

python src/features.py

python src/train\_models.py

python src/evaluate\_and\_register.py

streamlit run app.py

mlflow ui --backend-store-uri sqlite:///mlflow.db

```



\## Results (test set)



| Task | Model | Key metrics |

|------|-------|-------------|

| Classification (`Good\_Investment`) | Logistic regression | accuracy 0.861, F1 0.856, ROC AUC 0.950 |

| | Random forest | accuracy 0.996, F1 0.996, ROC AUC 0.9995 |

| | \*\*Hist. gradient boosting (best)\*\* | accuracy 0.999, F1 0.999, ROC AUC 1.000 |

| Regression (`Future\_Price\_5Y`, lakhs) | Ridge | RMSE 9.51, MAE 6.72, R2 0.998 |

| | \*\*Random forest (best)\*\* | RMSE 0.36, MAE 0.22, R2 1.000 |

| | Hist. gradient boosting | RMSE 1.21, MAE 0.97, R2 1.000 |



Confusion matrix: `reports/confusion\_matrix.png`. Feature importance: `reports/feature\_importance\_\*.csv`.



\## Use cases



\- Investors compare a listing against its city average and see a 5-year price forecast.

\- Buyers filter the market by state, BHK, price and size to find high-return areas.

\- Real estate platforms can automate investment screening of new listings.



\## Limitation



The targets are derived from price, size and location, and those are also model inputs. The near-perfect scores therefore show that the models recovered the target rules, not that they predict real market outcomes. Treat the app as a demo, not financial advice.

