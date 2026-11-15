import pandas as pd


def load_data(path):
    """Read the churn CSV file and return it as a DataFrame."""
    df = pd.read_csv(path)
    return df


def inspect_data(df):
    """Print a basic overview of the DataFrame."""
    print("=== FIRST 5 ROWS ===")
    print(df.head())

    print("\n=== SHAPE (rows, columns) ===")
    print(df.shape)

    print("\n=== COLUMN NAMES ===")
    print(df.columns.tolist())

    print("\n=== INFO ===")
    df.info()

    print("\n=== STATISTICAL SUMMARY (numeric columns) ===")
    print(df.describe())

    print("\n=== MISSING VALUES PER COLUMN ===")
    print(df.isnull().sum())

    print("\n=== TARGET COUNTS ===")
    print(df["Churn"].value_counts())


if __name__ == "__main__":
    data = load_data("data/Telco-Customer-Churn.csv")
    inspect_data(data)
