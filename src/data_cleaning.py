import pandas as pd


def standardize_column_names(df):
    """
    Standardize column names by removing extra spaces and converting them to lowercase.
    """
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    return df


def get_initial_overview(df, id_column=None):
    """
    Return a basic overview of the dataset: rows, columns, missing values,
    duplicated rows, and duplicated IDs if an ID column is provided.
    """
    overview = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": df.isna().sum().sum(),
        "duplicated_rows": df.duplicated().sum()
    }

    if id_column and id_column in df.columns:
        overview[f"duplicated_{id_column}"] = df[id_column].duplicated().sum()

    return pd.DataFrame.from_dict(overview, orient="index", columns=["count"])


def create_target_summary(df, target_column):
    """
    Return count and percentage distribution for the target variable.
    """
    target_summary = pd.DataFrame({
        "count": df[target_column].value_counts(),
        "percentage": df[target_column].value_counts(normalize=True).mul(100).round(2)
    })

    return target_summary


def create_sanity_checks(df):
    """
    Return key sanity checks used during the hotel reservation cleaning process.
    """
    sanity_checks = {
        "zero_adults": df[df["no_of_adults"] == 0].shape[0],
        "zero_price": df[df["avg_price_per_room"] == 0].shape[0],
        "children_5_or_more": df[df["no_of_children"] >= 5].shape[0],
        "lead_time_over_365": df[df["lead_time"] > 365].shape[0],
        "previous_cancellations_over_5": df[df["no_of_previous_cancellations"] > 5].shape[0],
        "previous_not_canceled_over_20": df[df["no_of_previous_bookings_not_canceled"] > 20].shape[0]
    }

    sanity_checks_df = pd.DataFrame.from_dict(sanity_checks, orient="index", columns=["count"])
    sanity_checks_df["percentage"] = (sanity_checks_df["count"] / len(df) * 100).round(3)

    return sanity_checks_df


def save_clean_dataset(df, output_path):
    """
    Save the cleaned dataset as a CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Clean dataset saved to: {output_path}")