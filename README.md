# Golf Course Demand Prediction

## Overview

Predicting daily golf course crowdedness (0.0–1.0) using weather and calendar features.

We compare four regression methods to find the best predictor of course demand:
- **Linear Regression** — baseline model
- **Lasso Regression** — feature selection via regularization
- **Random Forest** — non-linear ensemble method
- **Gradient Boosting** — sequential ensemble method (best performer, Test R² ≈ 0.90)

## Dataset

[Golf Play Dataset Extended](https://www.kaggle.com/datasets/samybaladram/golf-play-extended) by Samy Baladram.

## Project Structure
```
golf-demand-prediction/
├── golf_dataset/
│   ├── golf_dataset_wide_format.csv
│   └── split_sets/
│       ├── training_data.csv
│       └── test_data.csv
├── data_prep.py           # data loading, cleaning, encoding
├── model_linear.py        # linear regression
├── model_lasso.py         # lasso regression
├── model_rf.py            # random forest
├── model_gb.py            # gradient boosting
├── app.py                 # Streamlit web app
├── data_splits.ipynb      # train/test split generation
├── data_exploration.ipynb # EDA and boxplots
└── main.ipynb             # Original "draftbook"
```
## Setup

Install dependencies:

```
pip install pandas numpy scikit-learn streamlit matplotlib
```

## Usage

### Run individual models
```
python model_linear.py
python model_lasso.py
python model_rf.py
python model_gb.py
```

Each script prints train R², test R², and cross-validation scores.

### Run the web app
```
streamlit run app.py
```

The app lets you select weather and calendar conditions, choose a model, and predict crowdedness. It also displays a model comparison table with R² and cross-validation results.

## Key Findings

- **Humidity**, **Weekday**, and **Temperature** are the three strongest predictors across all models
- **Season** adds almost no predictive value — weather features already capture seasonal effects
- **Lasso** found **Outlook_overcast** to be the only truly uninformative feature
- **Temperature** has a non-linear, inverted-U relationship with crowdedness — linear models can't capture this, which explains their lower performance
- **Gradient Boosting** achieved the best test R² (≈ 0.90) with the least overfitting among non-linear models

## Authors

Pedro Nogueira & Neel Awsare