from data_prep import load_and_clean
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_lasso(alpha=0.0001):
    df = load_and_clean()

    X = df.drop(columns=["Crowdedness"])
    y = df["Crowdedness"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = Lasso(alpha=alpha)
    model.fit(X_train_scaled, y_train)

    return model, scaler, X_train.columns.tolist()


if __name__ == "__main__":
    df = load_and_clean()

    X = df.drop(columns=["Crowdedness"])
    y = df["Crowdedness"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("-------------------------------")

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

    print("-------------------------------")
    # print standard deviation of each feature
    for name, std in zip(X_train.columns, scaler.scale_):
        print(f"{name:<20} std = {std:.4f}")
