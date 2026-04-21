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

if __name__ == "__main__":
    df = load_and_clean()
    print(df.columns.tolist())
    print(df.shape)
