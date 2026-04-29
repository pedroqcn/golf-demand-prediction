from data_prep import load_and_clean
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split


def train_linear():
    df = load_and_clean()

    X = df.drop(columns=["Crowdedness"])
    y = df["Crowdedness"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    scores = {
        "train_r2": r2_score(y_train, model.predict(X_train)),
        "test_r2": r2_score(y_test, model.predict(X_test)),
    }

    return model, None, X_train.columns.tolist(), scores


if __name__ == "__main__":
    df = load_and_clean()

    X = df.drop(columns=["Crowdedness"])
    y = df["Crowdedness"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print(f"Train R^2: {r2_score(y_train, y_train_pred):.4f}")
    print(f"Test R^2: {r2_score(y_test, y_test_pred):.4f}")
    print(f"Test RMSE: {mean_squared_error(y_test, y_test_pred) ** 0.5:.4f}")
