import argparse
from pathlib import Path

import numpy as np

from .data import load_dataset
from .eda import build_summary, save_figures
from .preprocessing import build_preprocessor, split_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the NYC Airbnb EDA pipeline")
    parser.add_argument("--input", type=Path, help="Path to AB_NYC_2019.csv")
    parser.add_argument("--output", type=Path, default=Path("reports"))
    args = parser.parse_args()

    df = load_dataset(args.input)
    print(f"Dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Invalid prices removed: {(df['price'] <= 0).sum()}")

    summary = build_summary(df.loc[df["price"] > 0])
    args.output.mkdir(parents=True, exist_ok=True)
    for name, table in summary.items():
        table.to_csv(args.output / f"{name}.csv")
    save_figures(df.loc[df["price"] > 0], args.output / "figures")

    x_train, x_test, y_train, y_test = split_data(df)
    preprocessor = build_preprocessor()
    x_train_processed = preprocessor.fit_transform(x_train)
    x_test_processed = preprocessor.transform(x_test)
    print(f"Train/test: {x_train.shape[0]:,}/{x_test.shape[0]:,} rows")
    print(f"Processed shapes: {x_train_processed.shape}/{x_test_processed.shape}")
    print(f"Finite values: {np.isfinite(x_train_processed).all() and np.isfinite(x_test_processed).all()}")
    print(f"Target medians: {y_train.median():.2f}/{y_test.median():.2f}")


if __name__ == "__main__":
    main()
