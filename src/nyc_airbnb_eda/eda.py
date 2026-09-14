from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def build_summary(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return the main tabular outputs from the exploratory analysis."""
    numeric = df.select_dtypes(include="number").drop(columns=["id", "host_id"])
    missing = pd.DataFrame({
        "missing": df.isna().sum(),
        "percent": (100 * df.isna().mean()).round(2),
    }).sort_values("missing", ascending=False)
    room_stats = df.groupby("room_type")["price"].agg(
        count="size", mean="mean", median="median"
    ).sort_values("median", ascending=False)
    group_stats = df.groupby("neighbourhood_group")["price"].agg(
        count="size", mean="mean", median="median"
    ).sort_values("median", ascending=False)
    return {
        "descriptive_statistics": numeric.describe().T,
        "missing_values": missing,
        "room_type_price": room_stats,
        "borough_price": group_stats,
    }


def save_figures(df: pd.DataFrame, output_dir: str | Path) -> None:
    """Save the compact visualizations used in the notebook."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    sns.histplot(df["price"], bins=50, kde=True, ax=axes[0])
    axes[0].set(xlabel="Price (USD)", xlim=(0, 1000), title="Price distribution")
    sns.countplot(data=df, x="room_type", order=df["room_type"].value_counts().index, ax=axes[1])
    axes[1].set(title="Listings by room type", xlabel="", ylabel="Count")
    sns.countplot(
        data=df,
        x="neighbourhood_group",
        order=df["neighbourhood_group"].value_counts().index,
        ax=axes[2],
    )
    axes[2].set(title="Listings by borough", xlabel="", ylabel="Count")
    fig.savefig(output_path / "overview.png", dpi=150)
    plt.close(fig)

    numeric = df.select_dtypes(include="number").drop(columns=["id", "host_id"])
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Numeric correlation matrix")
    fig.savefig(output_path / "correlations.png", dpi=150)
    plt.close(fig)
