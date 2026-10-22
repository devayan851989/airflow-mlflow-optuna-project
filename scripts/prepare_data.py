#!/usr/bin/env python3
"""Generate reproducible synthetic housing data for the tutorial DAGs."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SAMPLE_COUNT = 2000
RANDOM_SEED = 42


def make_housing_dataset(sample_count: int = SAMPLE_COUNT, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Build a small synthetic dataset with a deterministic target formula."""
    rng = np.random.default_rng(seed)
    area = rng.normal(1500, 400, sample_count).clip(400, 5000)
    bedrooms = rng.integers(1, 7, sample_count)
    bathrooms = (bedrooms * rng.uniform(0.5, 1.0, sample_count)).astype(int).clip(1, 5)
    house_age = rng.integers(0, 50, sample_count)
    distance_km = rng.exponential(10, sample_count).clip(0.5, 80)
    garage_spots = rng.integers(0, 4, sample_count).astype(float)
    garage_spots[rng.random(sample_count) < 0.05] = np.nan
    noise = rng.normal(0, 15_000, sample_count)
    price = (
        area * 150
        + bedrooms * 8_000
        + bathrooms * 5_000
        - house_age * 1_200
        - distance_km * 3_000
        + noise
    ).clip(50_000, 2_000_000)

    return pd.DataFrame(
        {
            "area_sqft": area.round(1),
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "house_age": house_age,
            "distance_km": distance_km.round(2),
            "price_usd": price.round(2),
            "garage_spots": garage_spots,
        }
    )


def prepare_dataset() -> Path:
    """Write raw CSV and feature-engineered Parquet data to the project."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = RAW_DIR / "housing_raw.csv"
    processed_path = PROCESSED_DIR / "housing_clean.parquet"
    raw_frame = make_housing_dataset()
    raw_frame.to_csv(raw_path, index=False)
    frame = raw_frame.copy()

    frame["garage_spots"] = frame["garage_spots"].fillna(frame["garage_spots"].median())
    frame["bed_bath_ratio"] = (frame["bedrooms"] / frame["bathrooms"].clip(lower=1)).round(2)
    frame["is_new_house"] = (frame["house_age"] < 5).astype(int)
    frame.to_parquet(processed_path, index=False)

    metadata = {
        "rows": int(len(frame)),
        "columns": list(raw_frame.columns),
        "seed": RANDOM_SEED,
        "source": "deterministic_synthetic_generator",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (RAW_DIR / "housing_raw_meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {len(frame)} rows to {processed_path}")
    return processed_path


if __name__ == "__main__":
    prepare_dataset()