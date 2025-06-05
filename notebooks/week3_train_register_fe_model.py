"""Training notebook for hotel reservation cancellation prediction using feature engineering.

This notebook trains and registers a LightGBM classification model for predicting
hotel reservation cancellations using Databricks Feature Engineering.

This notebook is designed to be run in VSCode.
"""

# COMMAND ----------
import os

import mlflow
from dotenv import load_dotenv
from marvelous.common import is_databricks
from pyspark.sql import SparkSession
from pathlib import Path

from hotel_reservations.config import ProjectConfig, Tags
from hotel_reservations.models.feature_lookup_model import FeatureLookUpModel

# COMMAND ----------
# If you have DEFAULT profile and are logged in with DEFAULT profile,
# skip these lines

if not is_databricks():
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")

config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")
spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": "abcd12345", "branch": "week3"})

# COMMAND ----------
# Initialize model with the config
fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)

# COMMAND ----------
# Create feature table
fe_model.create_feature_table()

# COMMAND ----------
# Define feature function for days until arrival
fe_model.define_feature_function()

# COMMAND ----------
# Load data
fe_model.load_data()

# COMMAND ----------
# Perform feature engineering
fe_model.feature_engineering()

# COMMAND ----------
# Train the model
fe_model.train()

# COMMAND ----------
# Register the model
fe_model.register_model()

# COMMAND ----------
# Let's run prediction on the test set using the latest model
# Load test set from Delta table
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(10)

# Drop feature lookup columns and target for prediction
# Keep only the features that are not looked up from the feature table
feature_columns = [col for col in config.num_features + config.cat_features
                   if col not in ["avg_price_per_room", "no_of_special_requests", "no_of_previous_cancellations"]]
feature_columns.append("Booking_ID")  # Include ID for feature lookup

X_test = test_set.select(*feature_columns)

# COMMAND ----------
# Ensure proper data types for the test data
from pyspark.sql.functions import col

X_test = X_test.withColumn("no_of_adults", col("no_of_adults").cast("int")) \
               .withColumn("no_of_children", col("no_of_children").cast("int")) \
               .withColumn("no_of_weekend_nights", col("no_of_weekend_nights").cast("int")) \
               .withColumn("no_of_week_nights", col("no_of_week_nights").cast("int")) \
               .withColumn("required_car_parking_space", col("required_car_parking_space").cast("int")) \
               .withColumn("lead_time", col("lead_time").cast("int")) \
               .withColumn("arrival_year", col("arrival_year").cast("int")) \
               .withColumn("arrival_month", col("arrival_month").cast("int")) \
               .withColumn("arrival_date", col("arrival_date").cast("int")) \
               .withColumn("repeated_guest", col("repeated_guest").cast("int")) \
               .withColumn("no_of_previous_bookings_not_canceled", col("no_of_previous_bookings_not_canceled").cast("int")) \
               .withColumn("total_nights", col("total_nights").cast("int")) \
               .withColumn("total_guests", col("total_guests").cast("int")) \
               .withColumn("price_per_person", col("price_per_person").cast("double")) \
               .withColumn("price_per_night", col("price_per_night").cast("double")) \
               .withColumn("with_children", col("with_children").cast("int")) \
               .withColumn("has_weekend_stay", col("has_weekend_stay").cast("int")) \
               .withColumn("booking_season", col("booking_season").cast("int"))

# COMMAND ----------
# Make predictions using the latest registered model
predictions = fe_model.load_latest_model_and_predict(X_test)

# Display predictions
predictions.select("Booking_ID", "prediction").show()

# COMMAND ----------
# Let's also check the feature table to see what was created
spark.sql(f"SELECT * FROM {config.catalog_name}.{config.schema_name}.hotel_features LIMIT 5").show()

# COMMAND ----------
# Test the feature function
spark.sql(f"""
    SELECT {config.catalog_name}.{config.schema_name}.calculate_days_until_arrival(2025, 12, 25) as days_until_christmas
""").show()

# COMMAND ----------
