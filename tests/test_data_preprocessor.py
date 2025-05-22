"""Test module for data preprocessing logic without PySpark dependencies.

This module tests the core preprocessing functionality by extracting and testing
the preprocessing logic separately from the DataProcessor class.
"""

from typing import Any
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split


@pytest.fixture
def mock_config() -> Mock:
    """Create a mock configuration for testing."""
    config = Mock()
    config.num_features = [
        "no_of_adults",
        "no_of_children",
        "no_of_weekend_nights",
        "no_of_week_nights",
        "required_car_parking_space",
        "lead_time",
        "arrival_year",
        "arrival_month",
        "arrival_date",
        "repeated_guest",
        "no_of_previous_cancellations",
        "no_of_previous_bookings_not_canceled",
        "avg_price_per_room",
        "no_of_special_requests",
    ]
    config.cat_features = ["type_of_meal_plan", "room_type_reserved", "market_segment_type"]
    config.target = "is_canceled"
    config.parameters = {"test_size": 0.2, "random_state": 42}
    return config


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample hotel reservations data for testing."""
    np.random.seed(42)
    n_samples = 100

    data = {
        "Booking_ID": [f"BOOK_{i:05d}" for i in range(n_samples)],
        "no_of_adults": np.random.randint(1, 5, n_samples),
        "no_of_children": np.random.randint(0, 3, n_samples),
        "no_of_weekend_nights": np.random.randint(0, 3, n_samples),
        "no_of_week_nights": np.random.randint(1, 8, n_samples),
        "type_of_meal_plan": np.random.choice(["Plan A", "Plan B", "Plan C"], n_samples),
        "required_car_parking_space": np.random.randint(0, 2, n_samples),
        "room_type_reserved": np.random.choice(["Room_Type_1", "Room_Type_2", "Room_Type_3"], n_samples),
        "lead_time": np.random.randint(0, 365, n_samples),
        "arrival_year": np.random.choice([2023, 2024], n_samples),
        "arrival_month": np.random.randint(1, 13, n_samples),
        "arrival_date": np.random.randint(1, 29, n_samples),
        "market_segment_type": np.random.choice(["Online", "Offline", "Corporate"], n_samples),
        "repeated_guest": np.random.randint(0, 2, n_samples),
        "no_of_previous_cancellations": np.random.randint(0, 5, n_samples),
        "no_of_previous_bookings_not_canceled": np.random.randint(0, 10, n_samples),
        "avg_price_per_room": np.random.uniform(50, 300, n_samples),
        "no_of_special_requests": np.random.randint(0, 6, n_samples),
        "booking_status": np.random.choice(["Not_Canceled", "Canceled"], n_samples),
    }

    return pd.DataFrame(data)


@pytest.fixture
def data_with_missing_values() -> dict:
    """Create sample data with missing values for testing."""
    data = {
        "Booking_ID": ["BOOK_001", "BOOK_002", "BOOK_003"],
        "no_of_adults": [2, 1, 3],
        "no_of_children": [None, 1, 0],
        "no_of_weekend_nights": [1, 2, 1],
        "no_of_week_nights": [3, 4, 2],
        "type_of_meal_plan": ["Plan A", "Plan B", "Plan A"],
        "required_car_parking_space": [1, 0, 1],
        "room_type_reserved": ["Room_Type_1", "Room_Type_2", "Room_Type_1"],
        "lead_time": [30, 45, 60],
        "arrival_year": [2024, 2024, 2024],
        "arrival_month": [6, 7, 8],
        "arrival_date": [15, 20, 10],
        "market_segment_type": ["Online", "Offline", "Corporate"],
        "repeated_guest": [0, 1, 0],
        "no_of_previous_cancellations": [None, 1, 2],
        "no_of_previous_bookings_not_canceled": [5, None, 3],
        "avg_price_per_room": [120.5, 150.0, 200.0],
        "no_of_special_requests": [2, 1, 3],
        "booking_status": ["Not_Canceled", "Canceled", "Not_Canceled"],
    }

    return pd.DataFrame(data)


def preprocess_data(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Extract the preprocessing logic from DataProcessor for independent testing."""
    # Handle numeric features
    num_features = config.num_features
    for col in num_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Handle categorical features
    cat_features = config.cat_features
    for cat_col in cat_features:
        df[cat_col] = df[cat_col].astype("category")

    # Feature engineering
    df["arrival_date_full"] = pd.to_datetime(
        {"year": df["arrival_year"], "month": df["arrival_month"], "day": df["arrival_date"]},
        errors="coerce",
    )

    df["total_nights"] = df["no_of_weekend_nights"] + df["no_of_week_nights"]
    df["total_guests"] = df["no_of_adults"] + df["no_of_children"]
    df["with_children"] = (df["no_of_children"] > 0).astype(int)
    df["price_per_person"] = df["avg_price_per_room"] / df["total_guests"].clip(lower=1)
    df["price_per_night"] = df["avg_price_per_room"] / df["total_nights"].clip(lower=1)
    df["booking_season"] = df["arrival_month"].apply(lambda x: (x - 1) // 3 + 1)
    df["has_weekend_stay"] = (df["no_of_weekend_nights"] > 0).astype(int)

    # Handle potential missing values
    df["no_of_children"].fillna(0, inplace=True)
    df["no_of_previous_cancellations"].fillna(0, inplace=True)
    df["no_of_previous_bookings_not_canceled"].fillna(0, inplace=True)

    # Create target variable
    df["is_canceled"] = (df["booking_status"] == "Canceled").astype(int)
    df["Booking_ID"] = df["Booking_ID"].astype(str)

    # Define additional features
    additional_num_features = [
        "total_nights",
        "total_guests",
        "price_per_person",
        "price_per_night",
        "with_children",
        "has_weekend_stay",
    ]

    additional_cat_features = ["booking_season"]

    # Update feature lists
    all_num_features = num_features + additional_num_features
    all_cat_features = cat_features + additional_cat_features

    # Define columns to keep
    relevant_columns = all_num_features + all_cat_features + [config.target, "Booking_ID", "arrival_date_full"]

    # Keep only relevant columns
    df = df[relevant_columns]

    return df


class TestPreprocessingLogic:
    """Test class for preprocessing logic without PySpark dependencies."""

    def test_basic_preprocessing(self, sample_data: pd.DataFrame, mock_config: Mock) -> None:
        """Test basic preprocessing functionality."""
        df = sample_data.copy()
        original_shape = df.shape

        processed_df = preprocess_data(df, mock_config)

        # Check that preprocessing ran without errors
        assert processed_df is not None
        assert len(processed_df) == original_shape[0]  # Same number of rows

        # Check that target variable was created
        assert "is_canceled" in processed_df.columns
        assert processed_df["is_canceled"].dtype in [np.int64, int]

        # Check that engineered features were created
        expected_features = [
            "arrival_date_full",
            "total_nights",
            "total_guests",
            "with_children",
            "price_per_person",
            "price_per_night",
            "booking_season",
            "has_weekend_stay",
        ]
        for feature in expected_features:
            assert feature in processed_df.columns

    def test_feature_engineering_calculations(self, sample_data: pd.DataFrame, mock_config: dict) -> None:
        """Test specific feature engineering calculations."""
        df = sample_data.copy()
        processed_df = preprocess_data(df, mock_config)

        # Test total_nights calculation
        original_weekend = df["no_of_weekend_nights"]
        original_week = df["no_of_week_nights"]
        expected_total_nights = original_weekend + original_week

        # Compare with processed data (need to align indices)
        actual_total_nights = processed_df["total_nights"]

        # Verify the calculation is correct
        assert (actual_total_nights == expected_total_nights).all()

        # Test total_guests calculation
        original_adults = df["no_of_adults"]
        original_children = df["no_of_children"]
        expected_total_guests = original_adults + original_children
        actual_total_guests = processed_df["total_guests"]

        assert (actual_total_guests == expected_total_guests).all()

        # Test with_children flag
        expected_with_children = (df["no_of_children"] > 0).astype(int)
        actual_with_children = processed_df["with_children"]

        assert (actual_with_children == expected_with_children).all()

        # Test booking_season calculation
        expected_seasons = df["arrival_month"].apply(lambda x: (x - 1) // 3 + 1)
        actual_seasons = processed_df["booking_season"]

        assert (actual_seasons == expected_seasons).all()

    def test_data_type_conversions(self, sample_data: pd.DataFrame, mock_config: dict[str, Any]) -> None:
        """Test that data types are correctly converted."""
        df = sample_data.copy()
        processed_df = preprocess_data(df, mock_config)

        # Check that numeric features are numeric
        for col in mock_config.num_features:
            if col in processed_df.columns:
                assert pd.api.types.is_numeric_dtype(processed_df[col])

        # Check that categorical features are categorical
        for col in mock_config.cat_features:
            if col in processed_df.columns:
                assert processed_df[col].dtype.name == "category"

        # Check that Booking_ID is string
        assert processed_df["Booking_ID"].dtype == "object"

    def test_missing_value_handling(self, data_with_missing_values: pd.DataFrame, mock_config: dict[str, Any]) -> None:
        """Test that preprocessing properly handles missing values."""
        df = data_with_missing_values.copy()
        processed_df = preprocess_data(df, mock_config)

        # Check that missing values in specific columns were filled with 0
        assert processed_df["no_of_children"].isna().sum() == 0
        assert processed_df["no_of_previous_cancellations"].isna().sum() == 0
        assert processed_df["no_of_previous_bookings_not_canceled"].isna().sum() == 0

    def test_edge_cases_zero_values(self, mock_config: dict[str, Any]) -> None:
        """Test handling of edge cases with zero values."""
        edge_case_data = pd.DataFrame(
            {
                "Booking_ID": ["BOOK_001", "BOOK_002"],
                "no_of_adults": [0, 1],  # Zero adults case
                "no_of_children": [0, 0],
                "no_of_weekend_nights": [0, 1],
                "no_of_week_nights": [0, 2],  # Zero week nights for first row
                "type_of_meal_plan": ["Plan A", "Plan B"],
                "required_car_parking_space": [0, 1],
                "room_type_reserved": ["Room_Type_1", "Room_Type_2"],
                "lead_time": [30, 45],
                "arrival_year": [2024, 2024],
                "arrival_month": [6, 7],
                "arrival_date": [15, 20],
                "market_segment_type": ["Online", "Offline"],
                "repeated_guest": [0, 1],
                "no_of_previous_cancellations": [0, 1],
                "no_of_previous_bookings_not_canceled": [5, 3],
                "avg_price_per_room": [120.5, 150.0],
                "no_of_special_requests": [2, 1],
                "booking_status": ["Not_Canceled", "Canceled"],
            }
        )

        processed_df = preprocess_data(edge_case_data, mock_config)

        # Verify that zero values were handled correctly (clipped to 1)
        assert (processed_df["price_per_person"] >= 0).all()
        assert (processed_df["price_per_night"] >= 0).all()
        assert not processed_df["price_per_person"].isna().any()
        assert not processed_df["price_per_night"].isna().any()

    def test_data_splitting_logic(self, sample_data: pd.DataFrame, mock_config: dict[str, Any]) -> None:
        """Test data splitting logic using the same approach as DataProcessor."""
        df = sample_data.copy()
        processed_df = preprocess_data(df, mock_config)

        test_size = mock_config.parameters.get("test_size", 0.2)
        random_state = mock_config.parameters.get("random_state", 42)

        train_set, test_set = train_test_split(
            processed_df, test_size=test_size, random_state=random_state, stratify=processed_df[mock_config.target]
        )

        # Check that split was performed
        assert isinstance(train_set, pd.DataFrame)
        assert isinstance(test_set, pd.DataFrame)

        # Check split ratios
        total_rows = len(processed_df)
        expected_test_size = int(total_rows * test_size)
        assert abs(len(test_set) - expected_test_size) <= 1  # Allow for rounding

        # Check that all rows are accounted for
        assert len(train_set) + len(test_set) == total_rows

        # Check that both sets have the same columns
        assert list(train_set.columns) == list(test_set.columns)

        # Check that target variable exists in both sets
        assert mock_config.target in train_set.columns
        assert mock_config.target in test_set.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
