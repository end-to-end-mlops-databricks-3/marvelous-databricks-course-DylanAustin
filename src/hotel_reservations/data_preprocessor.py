"""Aligned to deliverable 1.

1. create a Python package that defines preprocessing steps for your chosen dataset so that it can be used in a machine learning project.
2. create a feature branch in your personal repository, and create a PR to main for us to review your code.
3. submit your PR in pr-submissions channel.

# Data Source
https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset

Data preprocessing module for hotel reservations dataset.

This module is responsible for loading the data, performing exploratory data analysis (EDA), and preprocessing the data for further analysis.
"""

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

from hotel_reservations.config import ProjectConfig


class DataProcessor:
    """A class for preprocessing and managing DataFrame operations.

    This class handles data preprocessing, splitting, and saving to Databricks tables
    for the hotel cancellations dataset.
    """

    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig, spark: SparkSession) -> None:
        """Initialize DataProcessor with a DataFrame, configuration, and Spark session.

        :param pandas_df: Input DataFrame to process
        :param config: Project configuration loaded from YAML
        :param spark: Active Spark session
        """
        self.df = pandas_df  # Store the DataFrame as self.df
        self.config = config  # Store the configuration
        self.spark = spark

    def preprocess(self) -> None:
        """Preprocess the DataFrame stored in self.df.

        This method handles missing values, converts data types, and performs feature engineering
        for the hotel cancellations dataset.
        """
        # Handle numeric features
        num_features = self.config.num_features
        for col in num_features:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # Handle categorical features
        cat_features = self.config.cat_features
        for cat_col in cat_features:
            self.df[cat_col] = self.df[cat_col].astype("category")

        # Feature engineering
        # Create a single arrival date column for time-based analysis
        self.df["arrival_date_full"] = pd.to_datetime(
            {"year": self.df["arrival_year"], "month": self.df["arrival_month"], "day": self.df["arrival_date"]},
            errors="coerce",  # Optional: sets invalid dates to NaT instead of raising
        )

        # Calculate total nights of stay
        self.df["total_nights"] = self.df["no_of_weekend_nights"] + self.df["no_of_week_nights"]

        # Calculate total guests
        self.df["total_guests"] = self.df["no_of_adults"] + self.df["no_of_children"]

        # Flag for bookings with children
        self.df["with_children"] = (self.df["no_of_children"] > 0).astype(int)

        # Calculate price per person (with safety for zero guests)
        self.df["price_per_person"] = self.df["avg_price_per_room"] / self.df["total_guests"].clip(lower=1)

        # Calculate price per night (with safety for zero nights)
        self.df["price_per_night"] = self.df["avg_price_per_room"] / self.df["total_nights"].clip(lower=1)

        # Add booking season (quarterly)
        self.df["booking_season"] = self.df["arrival_month"].apply(lambda x: (x - 1) // 3 + 1)

        # Calculate days until weekend (people might cancel differently based on this)
        self.df["has_weekend_stay"] = (self.df["no_of_weekend_nights"] > 0).astype(int)

        # Handle potential missing values
        self.df["no_of_children"].fillna(0, inplace=True)
        self.df["no_of_previous_cancellations"].fillna(0, inplace=True)
        self.df["no_of_previous_bookings_not_canceled"].fillna(0, inplace=True)

        # Create target variable (1 for canceled, 0 for not canceled)
        self.df["is_canceled"] = (self.df["booking_status"] == "Canceled").astype(int)

        # Ensure Booking_ID is treated as string
        self.df["Booking_ID"] = self.df["Booking_ID"].astype(str)

        # Define additional features to add to the feature lists
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
        relevant_columns = all_num_features + all_cat_features + [self.config.target, "Booking_ID", "arrival_date_full"]

        # Keep only relevant columns
        self.df = self.df[relevant_columns]

        # Log preprocessing completion
        print(f"Preprocessing complete. DataFrame shape: {self.df.shape}")
        print(f"Numeric features: {all_num_features}")
        print(f"Categorical features: {all_cat_features}")

    def split_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the DataFrame (self.df) into training and test sets.

        Uses test_size and random_state from config parameters.
        :return: A tuple containing the training and test DataFrames.
        """
        test_size = self.config.parameters.get("test_size", 0.2)
        random_state = self.config.parameters.get("random_state", 42)

        train_set, test_set = train_test_split(
            self.df,
            test_size=test_size,
            random_state=random_state,
            stratify=self.df[self.config.target],  # Stratified split based on target variable
        )

        print(f"Data split complete. Train set: {train_set.shape}, Test set: {test_set.shape}")
        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame) -> None:
        """Save the train and test sets into Databricks tables.

        :param train_set: The training DataFrame to be saved.
        :param test_set: The test DataFrame to be saved.
        """
        # Add timestamp for tracking when the data was processed
        train_set_with_timestamp = self.spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = self.spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        # Save to Databricks tables
        catalog_path = f"{self.config.catalog_name}.{self.config.schema_name}"

        train_set_with_timestamp.write.mode("overwrite").saveAsTable(f"{catalog_path}.train_set")
        test_set_with_timestamp.write.mode("overwrite").saveAsTable(f"{catalog_path}.test_set")

        print(f"Data saved to tables: {catalog_path}.train_set and {catalog_path}.test_set")

    def enable_change_data_feed(self) -> None:
        """Enable Change Data Feed for train and test set tables.

        This method alters the tables to enable Change Data Feed functionality.
        """
        catalog_path = f"{self.config.catalog_name}.{self.config.schema_name}"

        self.spark.sql(f"ALTER TABLE {catalog_path}.train_set SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

        self.spark.sql(f"ALTER TABLE {catalog_path}.test_set SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

        print("Change Data Feed enabled for train and test set tables")
