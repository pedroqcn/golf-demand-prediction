import pandas as pd
import numpy as np

def load_and_clean():
    """
    Loads the data and cleans certain parameters for the model to work from

    Returns:
        The clean DataFrame
    """

    df = pd.read_csv("golf_dataset/golf_dataset_wide_format.csv")

    # drop play columns, date, and month
    df = df.drop(columns=df.filter(like="Play").columns)
    df = df.drop(columns=["Date", "Month"])

    # cyclic encoding for Weekday
    df["Weekday_sin"] = np.sin(2 * np.pi * df["Weekday"] / 7)
    df["Weekday_cos"] = np.cos(2 * np.pi * df["Weekday"] / 7)

    df = df.drop(columns=["Weekday"]) # the sine and cosine pair replaces the weekday column

    # one-hot encode Outlook and Season
    df = pd.get_dummies(df, columns=["Outlook", "Season"], dtype=int)

    return df

def get_season_temp_ranges():
    """
    Returns min and max temperature for each season
    before one-hot encoding.
    """
    df = pd.read_csv("golf_dataset/golf_dataset_wide_format.csv")

    ranges = (
        df.groupby("Season")["Temperature"]
        .agg(["min", "max"])
        .to_dict(orient="index")
    )

    return ranges

def get_season_humidity_ranges():
    """
    Returns min and max humidity for each season
    before one-hot encoding.
    """
    df = pd.read_csv("golf_dataset/golf_dataset_wide_format.csv")

    ranges = (
        df.groupby("Season")["Humidity"]
        .agg(["min", "max"])
        .to_dict(orient="index")
    )

    return ranges

if __name__ == "__main__":
    df = load_and_clean()
    print(df.columns.tolist())
    print(df.shape)
