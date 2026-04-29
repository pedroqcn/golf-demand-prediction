from data_prep import load_and_clean
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score


def train_linear():
    train, test = load_and_clean()

    X_train = train.drop(columns=["Crowdedness"])
    y_train = train["Crowdedness"]
    X_test = test.drop(columns=["Crowdedness"])
    y_test = test["Crowdedness"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")

    scores = {
        "train_r2": r2_score(y_train, model.predict(X_train)),
        "test_r2": r2_score(y_test, model.predict(X_test)),
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "cv_scores": cv_scores
    }

    return model, None, X_train.columns.tolist(), scores


if __name__ == "__main__":
    train, test = load_and_clean()

    X_train = train.drop(columns=["Crowdedness"])
    y_train = train["Crowdedness"]
    X_test = test.drop(columns=["Crowdedness"])
    y_test = test["Crowdedness"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print(f"Train R^2: {r2_score(y_train, y_train_pred):.4f}")
    print(f"Test R^2: {r2_score(y_test, y_test_pred):.4f}")
    print(f"Test RMSE: {mean_squared_error(y_test, y_test_pred) ** 0.5:.4f}")
    print(f"CV R^2 scores: {cv_scores}")
    print(f"CV R^2 mean:   {cv_scores.mean():.4f}")
    print(f"CV R^2 std:    {cv_scores.std():.4f}")
