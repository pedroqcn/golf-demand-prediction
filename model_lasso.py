from data_prep import load_and_clean
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

df = load_and_clean()

# split train and test sets chronologically
train = df[df["Date"] < "2023-01-01"]
test = df[df["Date"] >= "2023-01-01"]

X_train = train.drop(columns=["Crowdedness", "Date"])
y_train = train["Crowdedness"]

X_test = test.drop(columns=["Crowdedness", "Date"])
y_test = test["Crowdedness"]

# feature scaling for better evaluation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# range of alpha to find the best penalty strength
alphas = [0.0001, 0.001, 0.01, 0.1, 1.0]

for a in alphas:
    model = Lasso(alpha=a)
    model.fit(X_train_scaled, y_train)
    y_test_pred = model.predict(X_test_scaled)
    print(f"Alpha: {a:<10} Test R^2: {r2_score(y_test, y_test_pred):.4f}")

# train final model with best alpha
best_model = Lasso(alpha=0.0001)
best_model.fit(X_train_scaled, y_train)

print("-------------------------------")

# show coefficients
feature_names = X_train.columns.tolist()
for name, coef in zip(feature_names, best_model.coef_):
    print(f"{name:<20} {coef:+.4f}")