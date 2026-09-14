import pandas as pd

from nyc_airbnb_eda.preprocessing import build_preprocessor, prepare_target


def test_prepare_target_removes_non_positive_prices() -> None:
    data = pd.DataFrame({"price": [0, 10, 25]})
    result = prepare_target(data)
    assert result["price"].tolist() == [10, 25]


def test_preprocessor_handles_missing_reviews() -> None:
    data = pd.DataFrame({
        "reviews_per_month": [None, 1.0],
        "minimum_nights": [1, 2],
        "number_of_reviews": [0, 3],
        "calculated_host_listings_count": [1, 2],
        "latitude": [40.7, 40.8],
        "longitude": [-73.9, -73.8],
        "availability_365": [10, 20],
        "neighbourhood_group": ["Manhattan", "Brooklyn"],
        "neighbourhood": ["Chelsea", "Williamsburg"],
        "room_type": ["Private room", "Entire home/apt"],
    })
    transformed = build_preprocessor().fit_transform(data)
    assert transformed.shape[0] == 2
