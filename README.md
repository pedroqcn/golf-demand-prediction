# Golf Crowd Demand Predictor

## Setup

Install dependencies (if pip doesn't work, try pip3):

```
pip install pandas numpy scikit-learn streamlit
```

## Running individual models

Each model script prints its own evaluation when run directly:

```
python model_linear.py
python model_lasso.py
python model_rf.py
python model_gb.py
```

## Running the web app

Launch the Streamlit app to interactively predict crowdedness and compare models:

```
streamlit run app.py
```

The app opens in your browser at a `localhost`.
