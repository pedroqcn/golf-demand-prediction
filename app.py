import numpy as np
import pandas as pd
import streamlit as st

from model_lasso import train_lasso
from model_linear import train_linear
from model_rf import train_rf


@st.cache_resource
def get_models():
    return {
        "Lasso": train_lasso(),
        "Linear": train_linear(),
        "Random Forest": train_rf(),
    }


models = get_models()

st.title("Golf Crowdedness Predictor")

choice = st.radio("Model", ["Lasso", "Linear", "Random Forest"], horizontal=True)
model, scaler, feature_names = models[choice]

col1, col2 = st.columns(2)
with col1:
    weekday = st.selectbox(
        "Day of week",
        options=[0, 1, 2, 3, 4, 5, 6],
        format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x],
    )
    season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Fall"])
    holiday = st.checkbox("Holiday")

with col2:
    temperature = st.slider("Temperature (°C)", -10.0, 40.0, 20.0)
    humidity = st.slider("Humidity (%)", 0.0, 100.0, 50.0)
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

    outlook_col = f"Outlook_{outlook}"
    if outlook_col in row:
        row[outlook_col] = 1

    season_col = f"Season_{season}"
    if season_col in row:
        row[season_col] = 1

    X_input = pd.DataFrame([row])[feature_names]
    X_final = scaler.transform(X_input) if scaler is not None else X_input
    prediction = model.predict(X_final)[0]

    st.metric("Predicted Crowdedness", f"{prediction:.2f}")

st.divider()

if choice == "Random Forest":
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
