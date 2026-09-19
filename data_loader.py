"""
Data loading and exploratory distribution utilities for Email/SMS Spam Detection.
"""

from pathlib import Path
import pandas as pd


DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "SMSSpamCollection"


def load_dataset(filepath: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Loads the SMS Spam Collection dataset into a pandas DataFrame.

    Parameters
    ----------
    filepath : str | Path
        Path to the tab-delimited SMSSpamCollection file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['label', 'message', 'label_num', 'char_count', 'word_count']
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    # The file is tab-separated: label (ham/spam) \t message
    df = pd.read_csv(
        filepath,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8",
    )

    # Clean any whitespace in label
    df["label"] = df["label"].str.strip()

    # Binary numeric encoding: 0 = ham (legitimate), 1 = spam
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

    # Exploratory text length metrics
    df["char_count"] = df["message"].astype(str).apply(len)
    df["word_count"] = df["message"].astype(str).apply(lambda x: len(x.split()))

    return df


def get_class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes class counts and percentages.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a 'label' column.

    Returns
    -------
    pd.DataFrame
        Summary table with counts and percentages per class.
    """
    counts = df["label"].value_counts()
    percentages = df["label"].value_counts(normalize=True) * 100

    summary = pd.DataFrame({
        "Count": counts,
        "Percentage (%)": percentages.round(2)
    })
    summary.index.name = "Class"
    return summary


if __name__ == "__main__":
    df = load_dataset()
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nClass Distribution:")
    print(get_class_distribution(df))
    print("\nFirst 3 rows:")
    print(df[["label", "message", "char_count", "word_count"]].head(3))
