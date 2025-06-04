# Databricks notebook source
"""Training notebook for hotel reservation cancellation prediction using basic model.

This notebook trains and registers a LightGBM classification model for predicting
hotel reservation cancellations in Databricks environment.

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
from hotel_reservations.models.basic_model import BasicModel

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
tags = Tags(**{"git_sha": "abcd12345", "branch": "week2"})

# COMMAND ----------
# Initialize model with the config path
basic_model = BasicModel(config=config, tags=tags, spark=spark)

# COMMAND ----------
basic_model.load_data()
basic_model.prepare_features()

# COMMAND ----------
# Train the model
basic_model.train()

# COMMAND ----------
# Log the model to MLflow
basic_model.log_model()

# COMMAND ----------
run_id = mlflow.search_runs(
    experiment_names=["/Users/dylanaustin37@gmail.com/hotel-cancellations-basic"], filter_string="tags.branch='week2'"
).run_id[0]

model = mlflow.sklearn.load_model(f"runs:/{run_id}/lightgbm-pipeline-model")

# COMMAND ----------
# Retrieve dataset for the current run
basic_model.retrieve_current_run_dataset()

# COMMAND ----------
# Retrieve metadata for the current run
basic_model.retrieve_current_run_metadata()

# COMMAND ----------
# Register model
basic_model.register_model()

# COMMAND ----------
# Predict on the test set

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(10)

X_test = test_set.drop(config.target).toPandas()

predictions_df = basic_model.load_latest_model_and_predict(X_test)

# COMMAND ----------
