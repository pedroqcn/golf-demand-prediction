import pandas as pd
from data_prep import load_and_clean
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def train_gb():
    df = load_and_clean()

    X = df.drop(columns=["Crowdedness"])
    y = df["Crowdedness"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42,
    )
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

    model = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print(f"Train R2:\t{r2_score(y_train, y_train_pred):.4f}")
    print(f"Test R2:\t{r2_score(y_test, y_test_pred):.4f}")

    imp_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\n-------------------------------------")
    print(imp_df)