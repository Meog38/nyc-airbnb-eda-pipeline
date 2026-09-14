import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

REVIEW_FEATURES = ["reviews_per_month"]
SKEWED_FEATURES = [
    "minimum_nights",
    "number_of_reviews",
    "calculated_host_listings_count",
]
OTHER_NUMERIC_FEATURES = ["latitude", "longitude", "availability_365"]
CATEGORICAL_FEATURES = ["neighbourhood_group", "neighbourhood", "room_type"]


def prepare_target(df: pd.DataFrame) -> pd.DataFrame:
    """Remove records with an invalid regression target."""
    return df.loc[df["price"] > 0].copy()


def split_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data before fitting any transformation."""
    clean_df = prepare_target(df)
    features = clean_df.drop(columns=["price"])
    target = clean_df["price"]
    target_deciles = pd.qcut(target, q=10, labels=False, duplicates="drop")
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target_deciles,
    )


def build_preprocessor() -> ColumnTransformer:
    """Build the leakage-safe preprocessing pipeline used by the notebook."""
    review_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("log", FunctionTransformer(np.log1p)),
        ("scaler", StandardScaler()),
    ])
    skewed_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p)),
        ("scaler", StandardScaler()),
    ])
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=20,
                sparse_output=False,
            ),
        ),
    ])
    return ColumnTransformer([
        ("skewed", skewed_pipeline, SKEWED_FEATURES),
        ("review", review_pipeline, REVIEW_FEATURES),
        ("numeric", numeric_pipeline, OTHER_NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
