import pandas as pd
import numpy as np\

def load_and_clean():
    """
    Loads the data and cleans certain parameters for the model to work from

    Returns:
        The clean DataFrame
    """

    df = pd.read_csv("golf_dataset/golf_dataset_wide_format.csv")

    # drop play, date, and season
    df = df.drop(columns=df.filter(like="Play").columns)
    df = df.drop(columns=["Month"])

    # cyclic encoding for Weekday
    df["Weekday_sin"] = np.sin(2 * np.pi * df["Weekday"] / 7)
    df["Weekday_cos"] = np.cos(2 * np.pi * df["Weekday"] / 7)

    df = df.drop(columns=["Weekday"]) # the sine and cosine pair replaces the weekday column

    # cylic encoding for Months
  # month_map = {
 #       "Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3,
  #      "May": 4, "Jun": 5, "Jul": 6, "Aug": 7,
   #     "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11
  #  }
  #  df["Month_num"] = df["Month"].map(month_map)

    #df["Month_sin"] = np.sin(2 * np.pi * df["Month_num"] / 12)
  #  df["Month_cos"] = np.cos(2 * np.pi * df["Month_num"] / 12)

   # df = df.drop(columns=["Month", "Month_num"])

    # one-hot encode Outlook
    df = pd.get_dummies(df, columns=["Outlook", "Season"], dtype=int)

    return df

if __name__ == "__main__":
    df = load_and_clean()
    print(df.columns.tolist())
    print(df.shape)