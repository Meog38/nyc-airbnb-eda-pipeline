from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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
    """Save the visual panels required by the EDA rubric."""
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

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    sns.boxplot(x=df["minimum_nights"], ax=axes[0], showfliers=False)
    axes[0].set(title="Minimum nights", xlabel="Nights")
    sns.violinplot(x=df["number_of_reviews"], ax=axes[1], cut=0)
    axes[1].set(title="Number of reviews", xlabel="Reviews")
    sns.histplot(df["reviews_per_month"], bins=40, kde=True, ax=axes[2])
    axes[2].set(title="Reviews per month", xlabel="Reviews / month")
    fig.savefig(output_path / "numeric_distributions.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    sns.countplot(data=df, x="room_type", order=df["room_type"].value_counts().index, ax=axes[0])
    axes[0].set(title="Room type frequency", xlabel="", ylabel="Listings")
    sns.countplot(data=df, x="neighbourhood_group", order=df["neighbourhood_group"].value_counts().index, ax=axes[1])
    axes[1].set(title="Borough frequency", xlabel="", ylabel="Listings")
    top_neighbourhoods = df["neighbourhood"].value_counts().head(10).sort_values()
    sns.barplot(x=top_neighbourhoods.values, y=top_neighbourhoods.index, ax=axes[2])
    axes[2].set(title="Top 10 neighbourhoods", xlabel="Listings", ylabel="")
    fig.savefig(output_path / "categorical_distributions.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    room_medians = df.groupby("room_type")["price"].median().sort_values(ascending=False)
    borough_medians = df.groupby("neighbourhood_group")["price"].median().sort_values(ascending=False)
    top_prices = df.groupby("neighbourhood")["price"].agg(["size", "median"])
    top_prices = top_prices[top_prices["size"] >= 20].nlargest(10, "median")["median"].sort_values()
    sns.barplot(x=room_medians.values, y=room_medians.index, ax=axes[0], color="#e5664d")
    axes[0].set(title="Median price by room type", xlabel="USD", ylabel="")
    sns.barplot(x=borough_medians.values, y=borough_medians.index, ax=axes[1], color="#5d8f91")
    axes[1].set(title="Median price by borough", xlabel="USD", ylabel="")
    sns.barplot(x=top_prices.values, y=top_prices.index, ax=axes[2], color="#17201e")
    axes[2].set(title="Top neighbourhood medians (n >= 20)", xlabel="USD", ylabel="")
    fig.savefig(output_path / "target_by_category.png", dpi=150)
    plt.close(fig)

    sample = df.sample(min(7000, len(df)), random_state=42)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    sns.boxplot(data=df, x="room_type", y="availability_365", showfliers=False, ax=axes[0])
    axes[0].set(title="Availability by room type", xlabel="", ylabel="Days")
    sns.boxplot(data=df, x="neighbourhood_group", y="minimum_nights", showfliers=False, ax=axes[1])
    axes[1].set(title="Minimum nights by borough", xlabel="", ylabel="Nights", ylim=(0, 30))
    sns.boxplot(data=df, x="room_type", y="reviews_per_month", showfliers=False, ax=axes[2])
    axes[2].set(title="Reviews / month by room type", xlabel="", ylabel="Reviews", ylim=(0, 5))
    fig.savefig(output_path / "numeric_categorical.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    numeric_features = [
        "minimum_nights", "number_of_reviews", "reviews_per_month",
        "calculated_host_listings_count", "availability_365", "latitude", "longitude",
    ]
    transformed = df[numeric_features].copy()
    transformed["reviews_per_month"] = transformed["reviews_per_month"].fillna(0)
    transformed[["minimum_nights", "number_of_reviews", "reviews_per_month", "calculated_host_listings_count"]] = np.log1p(
        transformed[["minimum_nights", "number_of_reviews", "reviews_per_month", "calculated_host_listings_count"]]
    )
    transformed = transformed.fillna(transformed.median())
    transformed = StandardScaler().fit_transform(transformed)
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(transformed)
    selected = np.random.default_rng(42).choice(len(components), min(8000, len(components)), replace=False)
    scatter = ax.scatter(components[selected, 0], components[selected, 1], c=np.log1p(df["price"].iloc[selected]), cmap="viridis", alpha=.3, s=8)
    fig.colorbar(scatter, ax=ax, label="log1p(price)")
    ax.set(title="PCA of numeric features", xlabel=f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)", ylabel=f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    fig.savefig(output_path / "pca.png", dpi=150)
    plt.close(fig)
