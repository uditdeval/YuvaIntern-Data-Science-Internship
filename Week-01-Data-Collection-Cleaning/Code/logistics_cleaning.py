"""
Week 1 - Logistics Data Collection and Cleaning
Author: Udit Deval
Dataset: UCI Bike Sharing (public transport-demand dataset) + local logistics CSV workflow

Install:
    pip install pandas numpy ucimlrepo

Run with the public UCI dataset:
    python logistics_cleaning.py --source uci

Run with your own shipment file:
    python logistics_cleaning.py --source csv --input shipments.csv

If no source is supplied, the script uses logistics_sample_raw.csv in this folder.
The small sample is illustrative and is NOT represented as real operational data.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def load_public_transport_dataset() -> pd.DataFrame:
    """Fetch UCI Bike Sharing dataset (ID 275) through the official repository client."""
    from ucimlrepo import fetch_ucirepo
    dataset = fetch_ucirepo(id=275)
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()
    df = pd.concat([features, targets], axis=1)
    return df

def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path)

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (df.columns.astype(str).str.strip().str.lower()
                  .str.replace(r"[^a-z0-9]+", "_", regex=True)
                  .str.strip("_"))
    return df

def parse_mixed_datetime(series: pd.Series) -> pd.Series:
    """Parse ISO-like year-first dates first, then day-first local formats."""
    values = series.astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    iso_mask = values.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}", na=False)
    parsed.loc[iso_mask] = pd.to_datetime(values.loc[iso_mask].str.replace("/", "-", regex=False), errors="coerce", format="mixed")
    remaining = ~iso_mask
    parsed.loc[remaining] = pd.to_datetime(values.loc[remaining], errors="coerce", dayfirst=True)
    return parsed

def clean_logistics_shipments(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean the shipment schema used in the task; preserve an audit summary."""
    df = standardize_column_names(df)
    required = ["shipment_id", "dispatch_time", "origin", "destination",
                "vehicle_type", "distance_km", "status"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    before_rows = len(df)
    exact_duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    # Trim whitespace and normalize text labels.
    for col in ["shipment_id", "origin", "destination", "vehicle_type", "status"]:
        df[col] = df[col].astype("string").str.strip()
    df["status"] = df["status"].str.lower().str.replace(r"\s+", " ", regex=True)
    status_map = {
        "delivered": "Delivered", "in transit": "In Transit",
        "pending": "Pending", "delayed": "Delayed"
    }
    df["status"] = df["status"].str.title().replace(status_map)

    # Parse dates; invalid values become NaT and are counted, not silently guessed.
    for col in ["dispatch_time", "delivery_time"]:
        if col in df.columns:
            df[col] = parse_mixed_datetime(df[col])

    # Convert numeric strings such as "2,400"; impossible negative distance -> missing.
    df["distance_km"] = (df["distance_km"].astype("string")
                         .str.replace(",", "", regex=False)
                         .str.strip())
    df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce")
    invalid_negative_distance = int((df["distance_km"] < 0).sum())
    df.loc[df["distance_km"] < 0, "distance_km"] = np.nan

    # Missing distance: keep row and flag it. Do not invent a route distance.
    df["distance_missing_flag"] = df["distance_km"].isna()
    # Missing/invalid dispatch timestamp is flagged for source-system correction.
    if "dispatch_time" in df.columns:
        df["dispatch_time_missing_flag"] = df["dispatch_time"].isna()

    # Delivery time earlier than dispatch is logically invalid; flag and null it.
    if "delivery_time" in df.columns and "dispatch_time" in df.columns:
        bad_order = (df["delivery_time"].notna() & df["dispatch_time"].notna()
                     & (df["delivery_time"] < df["dispatch_time"]))
        invalid_delivery_order = int(bad_order.sum())
        df.loc[bad_order, "delivery_time"] = pd.NaT
    else:
        invalid_delivery_order = 0

    report = {
        "rows_before": before_rows,
        "exact_duplicates_removed": exact_duplicates,
        "rows_after": len(df),
        "invalid_negative_distances": invalid_negative_distance,
        "invalid_delivery_before_dispatch": invalid_delivery_order,
        "missing_distance_after_cleaning": int(df["distance_km"].isna().sum()),
    }
    return df, report

def clean_uci_transport(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply safe generic checks to UCI Bike Sharing data; retain valid zero values."""
    df = standardize_column_names(df)
    before = len(df)
    duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    # The UCI date field is commonly named dteday.
    if "dteday" in df.columns:
        df["dteday"] = pd.to_datetime(df["dteday"], errors="coerce")
    # Convert object columns that are numeric in meaning, without forcing categories.
    for col in df.columns:
        if df[col].dtype == "object" and col not in {"dteday"}:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() >= max(1, int(0.8 * df[col].notna().sum())):
                df[col] = converted
    report = {
        "rows_before": before,
        "exact_duplicates_removed": duplicates,
        "rows_after": len(df),
        "missing_cells_after": int(df.isna().sum().sum()),
    }
    return df, report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["uci", "csv"], default="csv")
    parser.add_argument("--input", default="logistics_sample_raw.csv")
    parser.add_argument("--output", default="cleaned_shipments.csv")
    args = parser.parse_args()

    if args.source == "uci":
        raw = load_public_transport_dataset()
        cleaned, report = clean_uci_transport(raw)
        output = Path("cleaned_uci_bike_sharing.csv")
    else:
        raw = load_csv(args.input)
        cleaned, report = clean_logistics_shipments(raw)
        output = Path(args.output)

    cleaned.to_csv(output, index=False)
    print("CLEANING SUMMARY")
    for key, value in report.items():
        print(f"{key}: {value}")
    print(f"Saved cleaned dataset to: {output.resolve()}")
    print("\\nPreview:")
    print(cleaned.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
