from data_prep import load_and_clean
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

df = load_and_clean()

# split train and test sets chronologically
# more efficient, since we're learning from past data and predicting future demand
# -
# train on 2021-2022, test on 2023
train = df[df["Date"] < "2023-01-01"]
test = df[df["Date"] >= "2023-01-01"]

X_train = train.drop(columns=["Crowdedness", "Date"])
y_train = train["Crowdedness"]

X_test = test.drop(columns=["Crowdedness", "Date"])
y_test = test["Crowdedness"]

# train the model
model = LinearRegression()
model.fit(X_train, y_train)

# predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# evaluation
print(f"Train R^2: {r2_score(y_train, y_train_pred):.4f}")
print(f"Test R^2: {r2_score(y_test, y_test_pred):.4f}")
print(f"Test RMSE: {mean_squared_error(y_test, y_test_pred) ** 0.5:.4f}")