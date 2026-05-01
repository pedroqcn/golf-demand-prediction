import pandas as pd
import numpy as np

def load_and_clean():
    """
    Loads the data and cleans certain parameters for the model to work from

    Returns:
        The clean DataFrame
    """

    train = pd.read_csv("golf_dataset/split_sets/training_data.csv")
    test = pd.read_csv("golf_dataset/split_sets/test_data.csv")

    # combine temporarily train and test
    train["_split"] = "train"
    test["_split"] = "test"
    df = pd.concat([train, test], ignore_index=True)

    # drop play columns, date, and month
    df = df.drop(columns=df.filter(like="Play").columns)
    df = df.drop(columns=["Date", "Season"])

    month_map = {
        "Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4, "Jun": 5,
        "Jul": 6, "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11
    }
    df["Month"] = df["Month"].map(month_map)

    # cyclic encoding for Weekdays and Months
    df["Weekday_sin"] = np.sin(2 * np.pi * df["Weekday"] / 7)
    df["Weekday_cos"] = np.cos(2 * np.pi * df["Weekday"] / 7)
    df["Month_sin"]   = np.sin(2 * np.pi * df["Month"]   / 12)
    df["Month_cos"]   = np.cos(2 * np.pi * df["Month"]   / 12)
    df = df.drop(columns=["Weekday"]) # the sine and cosine pair replaces the weekday column
    df = df.drop(columns=["Month"])

    # one-hot encode Outlook
    df = pd.get_dummies(df, columns=["Outlook"], dtype=int)

    train = df[df["_split"] == "train"].drop(columns=["_split"])
    test = df[df["_split"] == "test"].drop(columns=["_split"])

    return train, test

def get_month_temp_ranges():
    """
    Returns min and max temperature for each month
    before one-hot encoding.
    """
    df = pd.read_csv("golf_dataset/golf_dataset_wide_format.csv")

    ranges = (
        df.groupby("Month")["Temperature"]
        .agg(["min", "max"])
        .to_dict(orient="index")
    )

    return ranges

def get_month_humidity_ranges():
    """
    Returns min and max humidity for each month
    before one-hot encoding.
    """
    df = pd.read_csv("golf_dataset/golf_dataset_wide_format.csv")

    ranges = (
        df.groupby("Month")["Humidity"]
        .agg(["min", "max"])
        .to_dict(orient="index")
    )

    return ranges

if __name__ == "__main__":
    train, test = load_and_clean()
    print(f"Train: {train.shape}, Test: {test.shape}")
    print(train.columns.tolist())