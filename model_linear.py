from data_prep import load_and_clean
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split


def train_linear():
    train, test = load_and_clean()

    X_train = train.drop(columns=["Crowdedness"])
    y_train = train["Crowdedness"]
    X_test = test.drop(columns=["Crowdedness"])
    y_test = test["Crowdedness"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    scores = {
        "train_r2": r2_score(y_train, model.predict(X_train)),
        "test_r2": r2_score(y_test, model.predict(X_test)),
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

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print(f"Train R^2: {r2_score(y_train, y_train_pred):.4f}")
    print(f"Test R^2: {r2_score(y_test, y_test_pred):.4f}")
    print(f"Test RMSE: {mean_squared_error(y_test, y_test_pred) ** 0.5:.4f}")
