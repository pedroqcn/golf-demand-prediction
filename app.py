import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from data_prep import load_and_clean, get_month_temp_ranges, get_month_humidity_ranges
from model_lasso import train_lasso
from model_linear import train_linear
from model_rf import train_rf
from model_gb import train_gb

MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

@st.cache_resource
def get_models():
    return {
        "Lasso": train_lasso(),
        "Linear": train_linear(),
        "Random Forest": train_rf(),
        "Gradient Boosting": train_gb(),
    }

# Initialize models
models = get_models()
month_temp_ranges = get_month_temp_ranges()
month_humidity_ranges = get_month_humidity_ranges()

st.title("Golf Crowdedness Predictor")

choice = st.radio("Model", ["Lasso", "Linear", "Random Forest", "Gradient Boosting"], horizontal=True)
model, scaler, feature_names, scores = models[choice]

st.subheader("Model Accuracy Comparison")
comparison_data = {}
for name in ["Lasso", "Linear", "Random Forest", "Gradient Boosting"]:
    s = models[name][3]
    comparison_data[name] = {
        "Train R²": s["train_r2"],
        "Test R²": s["test_r2"],
        "CV R² Mean": s["cv_mean"],
        "CV R² Standard Deviation": s["cv_std"],
    }

comparison_df = pd.DataFrame(comparison_data)
st.dataframe(comparison_df.style.format("{:.4f}", na_rep="-"))

c1, c2 = st.columns(2)
c1.metric(f"{choice} Train R²", f"{scores['train_r2']:.4f}")
c2.metric(f"{choice} Test R²", f"{scores['test_r2']:.4f}")

col1, col2 = st.columns(2)
with col1:
    weekday = st.selectbox(
        "Day of week",
        options=[0, 1, 2, 3, 4, 5, 6],
        format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x],
    )
    month = st.selectbox("Month", MONTH_ORDER)
    holiday = st.checkbox("Holiday")

temp_min = float(month_temp_ranges[month]["min"])
temp_max = float(month_temp_ranges[month]["max"])
temp_default = float((temp_min + temp_max) / 2)

hum_min = float(month_humidity_ranges[month]["min"])
hum_max = float(month_humidity_ranges[month]["max"])
hum_default = float((hum_min + hum_max) / 2)

with col2:
    temperature = st.slider(
        "Temperature (°C)",
        min_value=temp_min,
        max_value=temp_max,
        value=temp_default,
    )
    humidity = st.slider(
        "Humidity (%)",
        min_value=hum_min,
        max_value=hum_max,
        value=hum_default,
    )
    windy = st.checkbox("Windy")
    outlook = st.selectbox("Outlook", ["sunny", "overcast", "rainy"])

if st.button("Predict"):
    row = {name: 0 for name in feature_names}
    row["Holiday"] = int(holiday)
    row["Temperature"] = temperature
    row["Humidity"] = humidity
    row["Windy"] = int(windy)
    row["Weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    row["Weekday_cos"] = np.cos(2 * np.pi * weekday / 7)

    month_num = MONTH_ORDER.index(month)
    row["Month_sin"] = np.sin(2 * np.pi * month_num / 12)
    row["Month_cos"] = np.cos(2 * np.pi * month_num / 12)

    outlook_col = f"Outlook_{outlook}"
    if outlook_col in row:
        row[outlook_col] = 1

    X_input = pd.DataFrame([row])[feature_names]
    X_final = scaler.transform(X_input) if scaler is not None else X_input
    prediction = model.predict(X_final)[0]

    st.metric("Predicted Crowdedness", f"{prediction:.2f}")

st.divider()

if choice in ("Random Forest", "Gradient Boosting"):
    st.subheader("Feature Importances")
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    })
    st.bar_chart(imp_df.set_index("feature"))
else:
    st.subheader(f"{choice} Feature Coefficients")
    coef_df = pd.DataFrame({"feature": feature_names, "coefficient": model.coef_})
    st.bar_chart(coef_df.set_index("feature"))
