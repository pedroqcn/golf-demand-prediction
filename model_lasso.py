from data_prep import load_and_clean
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def train_lasso(alpha=0.0001):
    train, test = load_and_clean()

    X_train = train.drop(columns=["Crowdedness"])
    y_train = train["Crowdedness"]
    X_test = test.drop(columns=["Crowdedness"])
    y_test = test["Crowdedness"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Lasso(alpha=alpha)
    model.fit(X_train_scaled, y_train)

    cv_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", Lasso(alpha=alpha))
    ])

    cv_scores = cross_val_score(cv_pipe, X_train, y_train, cv=5, scoring="r2")

    scores = {
        "train_r2": r2_score(y_train, model.predict(X_train_scaled)),
        "test_r2": r2_score(y_test, model.predict(X_test_scaled)),
        "cv_mean": None,
        "cv_std": None,
    }

    return model, scaler, X_train.columns.tolist(), scores


if __name__ == "__main__":
    train, test = load_and_clean()

    X_train = train.drop(columns=["Crowdedness"])
    y_train = train["Crowdedness"]
    X_test = test.drop(columns=["Crowdedness"])
    y_test = test["Crowdedness"]

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

    cv_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", Lasso(alpha=0.0001))
    ])
    cv_scores = cross_val_score(cv_pipe, X_train, y_train, cv=5, scoring="r2")

    print("-------------------------------")
    print(f"CV R^2 scores: {cv_scores}")
    print(f"CV R^2 mean:   {cv_scores.mean():.4f}")
    print(f"CV R^2 std:    {cv_scores.std():.4f}")
    print("-------------------------------")

    # show coefficients
    feature_names = X_train.columns.tolist()
    for name, coef in zip(feature_names, best_model.coef_):
        print(f"{name:<20} {coef:+.4f}")

    print("-------------------------------")
    # print standard deviation of each feature
    for name, std in zip(X_train.columns, scaler.scale_):
        print(f"{name:<20} std = {std:.4f}")
