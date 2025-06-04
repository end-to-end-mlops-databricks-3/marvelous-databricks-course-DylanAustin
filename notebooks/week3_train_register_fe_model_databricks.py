# Databricks notebook source
dbutils.library.restartPython()

# COMMAND ----------

"""Training notebook for hotel reservation cancellation prediction using feature engineering.

This notebook trains and registers a LightGBM classification model for predicting
hotel reservation cancellations using Databricks Feature Engineering.

This notebook is designed to be run in VSCode.
"""

# COMMAND ----------

# Install required packages
%pip install loguru databricks-feature-engineering databricks-sdk lightgbm

# COMMAND ----------

# Restart Python to ensure packages are loaded
dbutils.library.restartPython()

# COMMAND ----------

# Set up the Python path to include the src directory
import sys
import os

# Add the src directory to Python path
project_root = "/Workspace/Users/dylanaustin37@gmail.com/.bundle/marvelous-databricks-course-DylanAustin/dev/files"
src_path = os.path.join(project_root, "src")
sys.path.insert(0, src_path)

# COMMAND ----------

# Import libraries
import mlflow
from loguru import logger
from pyspark.sql import SparkSession

from hotel_reservations.config import ProjectConfig, Tags
from hotel_reservations.models.feature_lookup_model import FeatureLookUpModel

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
tags_dict = {"git_sha": "abcd12345", "branch": "week2"}
tags = Tags(**tags_dict)

config = ProjectConfig.from_yaml(config_path="../project_config.yml")

# COMMAND ----------

# Initialize model
fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)

# COMMAND ----------

# Create feature table
fe_model.create_feature_table()                                                                                      

# COMMAND ----------

# Define feature function
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

# Test prediction
#test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(10)

# Get all columns that were used during training (check the training set structure)
#training_columns = fe_model.training_df.columns.tolist()
#print(f"Training columns: {training_columns}")

# For prediction, we need to include all columns except the target
#prediction_columns = [col for col in training_columns if col != config.target]
#print(f"Prediction columns needed: {prediction_columns}")

# Select the required columns from test set
#X_test = test_set.select(*[col for col in prediction_columns if col != "days_until_arrival"])  # days_until_arrival will be computed by feature function


# COMMAND ----------

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(10)

# Get the feature columns that the model needs (excluding lookup features)
required_base_columns = []
for col in config.num_features + config.cat_features:
    if col not in ["avg_price_per_room", "no_of_special_requests", "no_of_previous_cancellations"]:
        required_base_columns.append(col)

# Add the Booking_ID for lookup and arrival_date_full (which was created during preprocessing)
required_base_columns.extend(["Booking_ID", "arrival_date_full"])

print(f"Required base columns: {required_base_columns}")

# Select only the required columns from test set
X_test = test_set.select(*required_base_columns)

# Add the days_until_arrival calculation using PySpark SQL functions
from pyspark.sql.functions import col, datediff, make_date, current_date, greatest, lit

X_test = X_test.withColumn(
    "days_until_arrival",
    greatest(
        lit(0),
        datediff(
            make_date(col("arrival_year"), col("arrival_month"), col("arrival_date")),
            current_date()
        )
    )
)

# Ensure proper data types
X_test = X_test.withColumn("arrival_year", col("arrival_year").cast("int")) \
               .withColumn("arrival_month", col("arrival_month").cast("int")) \
               .withColumn("arrival_date", col("arrival_date").cast("int")) \
               .withColumn("no_of_adults", col("no_of_adults").cast("long")) \
               .withColumn("no_of_children", col("no_of_children").cast("long")) \
               .withColumn("no_of_weekend_nights", col("no_of_weekend_nights").cast("long")) \
               .withColumn("no_of_week_nights", col("no_of_week_nights").cast("long")) \
               .withColumn("required_car_parking_space", col("required_car_parking_space").cast("long")) \
               .withColumn("lead_time", col("lead_time").cast("long")) \
               .withColumn("repeated_guest", col("repeated_guest").cast("long")) \
               .withColumn("no_of_previous_bookings_not_canceled", col("no_of_previous_bookings_not_canceled").cast("long")) \
               .withColumn("total_nights", col("total_nights").cast("long")) \
               .withColumn("total_guests", col("total_guests").cast("long")) \
               .withColumn("price_per_person", col("price_per_person").cast("double")) \
               .withColumn("price_per_night", col("price_per_night").cast("double")) \
               .withColumn("with_children", col("with_children").cast("int")) \
               .withColumn("has_weekend_stay", col("has_weekend_stay").cast("int")) \
               .withColumn("booking_season", col("booking_season").cast("long")) \
               .withColumn("days_until_arrival", col("days_until_arrival").cast("int"))

print("X_test schema after adding days_until_arrival:")
X_test.printSchema()
#print("Sample data:")
#X_test.show(3)


# COMMAND ----------

# Cell 14: Make predictions
predictions = fe_model.load_latest_model_and_predict(X_test)
predictions.select("Booking_ID", "prediction").show()
