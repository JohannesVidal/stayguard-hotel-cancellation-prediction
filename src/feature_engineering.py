import pandas as pd


def encode_target(df, target_column="booking_status"):
    """
    Encode booking status for binary classification.

    Not_Canceled = 0
    Canceled = 1
    """
    df = df.copy()
    df[target_column] = df[target_column].map({"Not_Canceled": 0, "Canceled": 1})
    return df


def split_features_target(df, target_column="booking_status", drop_columns=None):
    """
    Split dataframe into features X and target y.
    """
    if drop_columns is None:
        drop_columns = []

    X = df.drop(columns=drop_columns + [target_column])
    y = df[target_column].map({"Not_Canceled": 0, "Canceled": 1})

    return X, y


def identify_feature_types(X):
    """
    Identify categorical and numerical features.
    """
    categorical_features = X.select_dtypes(include="object").columns.tolist()
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    return categorical_features, numerical_features


def create_lead_time_groups(df):
    """
    Create lead time groups for exploratory analysis.
    """
    df = df.copy()

    df["lead_time_group"] = pd.cut(
        df["lead_time"],
        bins=[-1, 7, 30, 90, 180, 365, df["lead_time"].max()],
        labels=["0-7 days", "8-30 days", "31-90 days", "91-180 days", "181-365 days", "365+ days"]
    )

    return df


def create_price_groups(df):
    """
    Create average room price groups for exploratory analysis.
    """
    df = df.copy()

    df["price_group"] = pd.cut(
        df["avg_price_per_room"],
        bins=[-1, 0, 75, 100, 125, 150, df["avg_price_per_room"].max()],
        labels=["0", "1-75", "76-100", "101-125", "126-150", "150+"]
    )

    return df