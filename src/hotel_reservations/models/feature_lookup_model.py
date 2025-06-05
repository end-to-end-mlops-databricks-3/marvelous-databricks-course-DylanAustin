"""FeatureLookUp model implementation for hotel cancellation predictions."""

import contextlib
from datetime import datetime

import mlflow
import pandas as pd

try:
    from databricks import feature_engineering
    from databricks.feature_engineering import FeatureLookup  # FeatureFunction,
    from databricks.sdk import WorkspaceClient
except ImportError as e:
    print(f"Warning: Could not import Databricks feature engineering: {e}")
    print("Please ensure you have the databricks-feature-engineering package installed")
    print("Run: %pip install databricks-feature-engineering databricks-sdk")
    raise
from lightgbm import LGBMClassifier
from loguru import logger
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from hotel_reservations.config import ProjectConfig, Tags


class FeatureLookUpModel:
    """A class to manage FeatureLookupModel for hotel cancellation prediction."""

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession) -> None:
        """Initialize the model with project configuration."""
        self.config = config
        self.spark = spark
        self.workspace = WorkspaceClient()
        self.fe = feature_engineering.FeatureEngineeringClient()

        # Extract settings from the config
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name

        # Define table names and function name
        self.feature_table_name = f"{self.catalog_name}.{self.schema_name}.hotel_features"
        self.function_name = f"{self.catalog_name}.{self.schema_name}.calculate_days_until_arrival"

        # MLflow configuration
        self.experiment_name = self.config.experiment_name_custom
        self.tags = tags.dict()

    def create_feature_table(self) -> None:
        """Create or update the hotel_features table and populate it.

        This table stores features related to hotel reservations for feature lookup.
        We'll store key features that might be looked up from external systems.
        """
        # Create the feature table with hotel-specific features
        self.spark.sql(f"""
        CREATE OR REPLACE TABLE {self.feature_table_name}
        (Booking_ID STRING NOT NULL,
         avg_price_per_room DOUBLE,
         no_of_special_requests INT,
         no_of_previous_cancellations INT);
        """)

        with contextlib.suppress(Exception):
            # Primary key constraint might already exist
            self.spark.sql(f"ALTER TABLE {self.feature_table_name} ADD CONSTRAINT hotel_pk PRIMARY KEY(Booking_ID);")

        self.spark.sql(f"ALTER TABLE {self.feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

        # Populate with training data
        self.spark.sql(f"""
            INSERT INTO {self.feature_table_name}
            SELECT Booking_ID, avg_price_per_room, no_of_special_requests, no_of_previous_cancellations
            FROM {self.catalog_name}.{self.schema_name}.train_set
        """)

        # Populate with test data
        self.spark.sql(f"""
            INSERT INTO {self.feature_table_name}
            SELECT Booking_ID, avg_price_per_room, no_of_special_requests, no_of_previous_cancellations
            FROM {self.catalog_name}.{self.schema_name}.test_set
        """)

        logger.info("✅ Feature table created and populated.")

    def define_feature_function(self) -> None:
        """Define a function to calculate days until arrival.

        This method is kept for interface compatibility but doesn't create a UDF
        since we'll calculate days_until_arrival directly in feature_engineering.
        """
        logger.info("✅ Skipping UDF creation - will calculate days_until_arrival directly in feature_engineering.")

    def load_data(self) -> None:
        """Load training and testing data from Delta tables.

        Drops specified columns that will be looked up from feature table.
        """
        # Load data and drop columns that will be looked up from feature table
        self.train_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set").drop(
            "avg_price_per_room", "no_of_special_requests", "no_of_previous_cancellations"
        )
        self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set").toPandas()

        # Ensure proper data types
        self.train_set = self.train_set.withColumn("arrival_year", self.train_set["arrival_year"].cast("int"))
        self.train_set = self.train_set.withColumn("arrival_month", self.train_set["arrival_month"].cast("int"))
        self.train_set = self.train_set.withColumn("arrival_date", self.train_set["arrival_date"].cast("int"))
        self.train_set = self.train_set.withColumn("Booking_ID", self.train_set["Booking_ID"].cast("string"))

        logger.info("✅ Data successfully loaded.")

    def feature_engineering(self) -> None:
        """Perform feature engineering by linking data with feature tables.

        Creates a training set using FeatureLookup and direct calculation instead of FeatureFunction.
        """
        from pyspark.sql.functions import col, current_date, datediff, greatest, lit, make_date

        # Calculate days_until_arrival directly in the DataFrame using PySpark functions
        train_set_with_days = self.train_set.withColumn(
            "days_until_arrival",
            greatest(
                lit(0),
                datediff(make_date(col("arrival_year"), col("arrival_month"), col("arrival_date")), current_date()),
            ),
        )

        self.training_set = self.fe.create_training_set(
            df=train_set_with_days,
            label=self.target,
            feature_lookups=[
                FeatureLookup(
                    table_name=self.feature_table_name,
                    feature_names=["avg_price_per_room", "no_of_special_requests", "no_of_previous_cancellations"],
                    lookup_key="Booking_ID",
                ),
                # No FeatureFunction - we calculated days_until_arrival directly
            ],
            exclude_columns=["update_timestamp_utc"],
        )

        self.training_df = self.training_set.load_df().toPandas()

        # Calculate days_until_arrival for test set manually
        today = datetime.now().date()

        def calc_days_until_arrival(row: pd.Series) -> int:
            try:
                arrival_date_obj = datetime(
                    int(row["arrival_year"]), int(row["arrival_month"]), int(row["arrival_date"])
                ).date()
                days_diff = (arrival_date_obj - today).days
                return max(0, days_diff)
            except (ValueError, TypeError, KeyError, OverflowError):
                return 0

        self.test_set["days_until_arrival"] = self.test_set.apply(calc_days_until_arrival, axis=1)

        # Build feature columns properly
        feature_columns = []

        # Add numeric features (excluding lookup features)
        for feature_col in self.num_features:
            if (
                feature_col not in ["avg_price_per_room", "no_of_special_requests", "no_of_previous_cancellations"]
                and feature_col in self.training_df.columns
            ):
                feature_columns.append(feature_col)

        # Add categorical features
        for feature_col in self.cat_features:
            if feature_col in self.training_df.columns:
                feature_columns.append(feature_col)

        # Add lookup features
        feature_columns.extend(["avg_price_per_room", "no_of_special_requests", "no_of_previous_cancellations"])

        # Add computed feature
        feature_columns.append("days_until_arrival")

        # Remove any non-feature columns
        feature_columns = [
            feature_col
            for feature_col in feature_columns
            if feature_col in self.training_df.columns and feature_col not in ["Booking_ID", "arrival_date_full"]
        ]

        logger.info(f"Selected feature columns: {feature_columns}")

        self.X_train = self.training_df[feature_columns]
        self.y_train = self.training_df[self.target]
        self.X_test = self.test_set[feature_columns]
        self.y_test = self.test_set[self.target]

        logger.info("✅ Feature engineering completed.")
        logger.info(f"Training shape: {self.X_train.shape}")
        logger.info(f"Feature columns: {list(self.X_train.columns)}")

    def train(self) -> None:
        """Train the model and log results to MLflow.

        Uses a pipeline with preprocessing and LightGBM classifier.
        """
        logger.info("🚀 Starting training...")

        # Filter categorical features that are actually present
        available_cat_features = [col for col in self.cat_features if col in self.X_train.columns]

        preprocessor = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), available_cat_features)],
            remainder="passthrough",
        )

        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", LGBMClassifier(**self.parameters))])

        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            pipeline.fit(self.X_train, self.y_train)
            y_pred = pipeline.predict(self.X_test)
            y_pred_proba = pipeline.predict_proba(self.X_test)[:, 1]

            # Calculate metrics for binary classification
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)

            logger.info(f"📊 Accuracy: {accuracy}")
            logger.info(f"📊 Precision: {precision}")
            logger.info(f"📊 Recall: {recall}")
            logger.info(f"📊 F1 Score: {f1}")
            logger.info(f"📊 ROC AUC: {roc_auc}")

            mlflow.log_param("model_type", "LightGBM Classification with Feature Engineering")
            mlflow.log_params(self.parameters)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc_auc)

            signature = infer_signature(self.X_train, y_pred)

            self.fe.log_model(
                model=pipeline,
                flavor=mlflow.sklearn,
                artifact_path="lightgbm-pipeline-model-fe",
                training_set=self.training_set,
                signature=signature,
            )

    def register_model(self) -> str:
        """Register the trained model to MLflow registry.

        Registers the model and sets alias to 'latest-model'.
        """
        model_name = f"{self.catalog_name}.{self.schema_name}.hotel_cancellation_model_fe"

        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/lightgbm-pipeline-model-fe",
            name=model_name,
            tags=self.tags,
        )

        # Fetch the latest version dynamically
        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=model_name,
            alias="latest-model",
            version=latest_version,
        )

        logger.info(f"✅ Model registered as version {latest_version} with alias 'latest-model'")

        return latest_version

    def load_latest_model_and_predict(self, X: DataFrame) -> DataFrame:
        """Load the trained model from MLflow using Feature Engineering Client and make predictions.

        Loads the model with the alias 'latest-model' and scores the batch.
        :param X: DataFrame containing the input features.
        :return: DataFrame containing the predictions.
        """
        model_name = f"{self.catalog_name}.{self.schema_name}.hotel_cancellation_model_fe"
        model_uri = f"models:/{model_name}@latest-model"

        predictions = self.fe.score_batch(model_uri=model_uri, df=X)
        return predictions

    def update_feature_table(self) -> None:
        """Update the hotel_features table with the latest records from train and test sets.

        Executes SQL queries to insert new records based on timestamp.
        """
        queries = [
            f"""
            WITH max_timestamp AS (
                SELECT MAX(update_timestamp_utc) AS max_update_timestamp
                FROM {self.config.catalog_name}.{self.config.schema_name}.train_set
            )
            INSERT INTO {self.feature_table_name}
            SELECT Booking_ID, avg_price_per_room, no_of_special_requests, no_of_previous_cancellations
            FROM {self.config.catalog_name}.{self.config.schema_name}.train_set
            WHERE update_timestamp_utc >= (SELECT max_update_timestamp FROM max_timestamp)
            """,
            f"""
            WITH max_timestamp AS (
                SELECT MAX(update_timestamp_utc) AS max_update_timestamp
                FROM {self.config.catalog_name}.{self.config.schema_name}.test_set
            )
            INSERT INTO {self.feature_table_name}
            SELECT Booking_ID, avg_price_per_room, no_of_special_requests, no_of_previous_cancellations
            FROM {self.config.catalog_name}.{self.config.schema_name}.test_set
            WHERE update_timestamp_utc >= (SELECT max_update_timestamp FROM max_timestamp)
            """,
        ]

        for query in queries:
            logger.info("Executing SQL update query...")
            self.spark.sql(query)
        logger.info("Hotel features table updated successfully.")

    def model_improved(self, test_set: DataFrame) -> bool:
        """Evaluate the model performance on the test set.

        Compares the current model with the latest registered model using F1 score.
        :param test_set: DataFrame containing the test data.
        :return: True if the current model performs better, False otherwise.
        """
        # Prepare test features (excluding lookup features and target)
        feature_columns = [
            col
            for col in self.num_features + self.cat_features
            if col not in ["avg_price_per_room", "no_of_special_requests", "no_of_previous_cancellations"]
        ]
        feature_columns.append("Booking_ID")  # Include ID for lookup

        X_test = test_set.select(*feature_columns)

        # Ensure proper data types
        X_test = X_test.withColumn("arrival_year", F.col("arrival_year").cast("int"))
        X_test = X_test.withColumn("arrival_month", F.col("arrival_month").cast("int"))
        X_test = X_test.withColumn("arrival_date", F.col("arrival_date").cast("int"))

        # Get predictions from latest model
        predictions_latest = self.load_latest_model_and_predict(X_test).withColumnRenamed(
            "prediction", "prediction_latest"
        )

        # Get predictions from current model
        current_model_uri = f"runs:/{self.run_id}/lightgbm-pipeline-model-fe"
        predictions_current = self.fe.score_batch(model_uri=current_model_uri, df=X_test).withColumnRenamed(
            "prediction", "prediction_current"
        )

        # Prepare test set with target
        test_set_target = test_set.select("Booking_ID", self.target)

        logger.info("Predictions are ready.")

        # Join the DataFrames on the 'Booking_ID' column
        df = test_set_target.join(predictions_current, on="Booking_ID").join(predictions_latest, on="Booking_ID")

        # Convert to pandas for easier metric calculation
        df_pandas = df.toPandas()

        # Calculate F1 scores for each model
        f1_current = f1_score(df_pandas[self.target], df_pandas["prediction_current"])
        f1_latest = f1_score(df_pandas[self.target], df_pandas["prediction_latest"])

        # Compare models based on F1 score
        logger.info(f"F1 Score for Current Model: {f1_current}")
        logger.info(f"F1 Score for Latest Model: {f1_latest}")

        if f1_current > f1_latest:
            logger.info("Current Model performs better.")
            return True
        else:
            logger.info("New Model performs worse.")
            return False
