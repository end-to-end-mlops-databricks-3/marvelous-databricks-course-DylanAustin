# Databricks notebook source
"""Training notebook for hotel reservation cancellation prediction using basic model.

This notebook trains and registers a LightGBM classification model for predicting
hotel reservation cancellations in Databricks environment.

This notebook is designed to be run in a Databricks.
"""
# COMMAND ----------

# Set up the Python path to include the src directory
import sys
import os

# Add the src directory to Python path
project_root = "/Workspace/Users/dylanaustin37@gmail.com/.bundle/marvelous-databricks-course-DylanAustin/dev/files"
src_path = os.path.join(project_root, "src")
sys.path.insert(0, src_path)

print(f"Added to Python path: {src_path}")
print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path[:3]}")  # Show first 3 entries

# COMMAND ----------

# Install required packages
%pip install lightgbm scikit-learn loguru


# COMMAND ----------

# Restart Python to ensure packages are loaded
dbutils.library.restartPython()

# COMMAND ----------

# Re-add the src directory to Python path after restart
import sys
import os

# Add the src directory to Python path
project_root = "/Workspace/Users/dylanaustin37@gmail.com/.bundle/marvelous-databricks-course-DylanAustin/dev/files"
src_path = os.path.join(project_root, "src")
sys.path.insert(0, src_path)

print(f"Added to Python path: {src_path}")

# COMMAND ----------

# Now import the modules
import mlflow
from pyspark.sql import SparkSession

try:
    from hotel_reservations.config import ProjectConfig, Tags
    from hotel_reservations.models.basic_model import BasicModel
    print("✅ Successfully imported hotel_reservations modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Available files in src directory:")
    if os.path.exists(src_path):
        for item in os.listdir(src_path):
            print(f"  - {item}")
            # Show what's inside hotel_reservations directory
            hr_path = os.path.join(src_path, "hotel_reservations")
            if os.path.exists(hr_path):
                print(f"  Contents of hotel_reservations:")
                for subitem in os.listdir(hr_path):
                    print(f"    - {subitem}")
    else:
        print(f"❌ src directory not found at: {src_path}")

# COMMAND ----------

# Configure MLflow for Databricks
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# Load configuration - adjust path as needed
config_path = os.path.join(project_root, "project_config.yml")
config = ProjectConfig.from_yaml(config_path=config_path, env="dev")

# Fix the experiment name to use proper Databricks format
config.experiment_name_basic = "/Users/dylanaustin37@gmail.com/hotel-cancellations-basic"

spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": "abcd12345", "branch": "week2"})

print(f"✅ Configuration loaded for environment: dev")
print(f"✅ Catalog: {config.catalog_name}, Schema: {config.schema_name}")
print(f"✅ Experiment name: {config.experiment_name_basic}")

# COMMAND ----------

# Initialize model with the config path
basic_model = BasicModel(config=config, tags=tags, spark=spark)

# COMMAND ----------

basic_model.load_data()
basic_model.prepare_features()

# COMMAND ----------

# Train model
basic_model.train()


# COMMAND ----------

# Log model (runs everything including MLflow logging)
basic_model.log_model()

# COMMAND ----------

# Search for runs (optional verification)
try:
    runs = mlflow.search_runs(
        experiment_names=[config.experiment_name_basic],
        filter_string="tags.branch='week2'"
    )
    if len(runs) > 0:
        run_id = runs.run_id[0]
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/lightgbm-pipeline-model")
        print(f"✅ Found run: {run_id}")
    else:
        print("⚠️ No runs found with branch='week2'")
except Exception as e:
    print(f"⚠️ Error searching runs: {e}")

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
