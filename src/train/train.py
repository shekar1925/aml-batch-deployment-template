import logging
import os
import traceback

import joblib
import numpy as np
from azureml.core import Run
from opencensus.ext.azure.log_exporter import AzureLogHandler
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

run = None
logger = None


def set_logger():
    global logger

    # Add the app insights logger to the python logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(AzureLogHandler())

    # Get pipeline information
    custom_dimensions = {
        "parent_run_id": run.parent.id,
        "step_id": run.id,
        "step_name": run.name,
        "experiment_name": run.experiment.name,
        "run_url": run.parent.get_portal_url(),
    }

    # Log pipeline information
    logger.info(custom_dimensions)


def load_data():
    # Retreive dataset
    dataset = run.input_datasets["InputDataset"]

    # Convert dataset to pandas dataframe
    df = dataset.to_pandas_dataframe()

    # Convert strings to float
    df = df.astype(
        {
            "age": np.float64,
            "height": np.float64,
            "weight": np.float64,
            "systolic": np.float64,
            "diastolic": np.float64,
            "cardiovascular_disease": np.float64,
        }
    )

    return df


def preprocess_data(df):
    # Remove missing values
    df.dropna(inplace=True)

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Remove records where height or weight is more than 6 std from mean
    df = df[(np.abs(stats.zscore(df.height)) < 6)]
    df = df[(np.abs(stats.zscore(df.weight)) < 6)]

    # Create feature for Body Mass Index (indicator of heart health)
    df["bmi"] = df.weight / (df.height / 100) ** 2

    return df


def train_model(df):
    # Define categorical features
    categorical_features = [
        "gender",
        "cholesterol",
        "glucose",
        "smoker",
        "alcoholic",
        "active",
    ]

    # Define numeric features
    numeric_features = ["age", "systolic", "diastolic", "bmi"]

    # Get model features / target
    X = df.drop(
        labels=["height", "weight", "cardiovascular_disease", "datetime"],
        axis=1,
        errors="ignore",
    )
    y = df.cardiovascular_disease

    # Convert data types of model features
    X[categorical_features] = X[categorical_features].astype(np.object)
    X[numeric_features] = X[numeric_features].astype(np.float64)

    # Define model pipeline
    scaler = StandardScaler()
    onehotencoder = OneHotEncoder(categories="auto")
    classifier = LogisticRegression(random_state=0, solver="liblinear")

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", scaler, numeric_features),
            ("categorical", onehotencoder, categorical_features),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", classifier)]
    )

    # Train / evaluate performance of logistic regression classifier
    cv_results = cross_validate(pipeline, X, y, cv=10, return_train_score=True)

    # Log average train / test accuracy
    for run_context in [run, run.parent]:
        run_context.log("train_acccuracy", round(cv_results["train_score"].mean(), 4))
        run_context.log("test_acccuracy", round(cv_results["test_score"].mean(), 4))

        # Log performance metrics for data
        for metric in cv_results.keys():
            run_context.log_row(
                "K-Fold CV Metrics",
                metric=metric.replace("_", " "),
                mean="{:.2%}".format(cv_results[metric].mean()),
                std="{:.2%}".format(cv_results[metric].std()),
            )

    # Fit model
    pipeline.fit(X, y)

    return pipeline


def main():
    try:
        global run

        # Retrieve current service context
        run = Run.get_context()

        # Set logger
        set_logger()

        # Load data, pre-process data, train and evaluate model
        df = load_data()
        df = preprocess_data(df)
        model = train_model(df)

        # Define model file name
        model_file_name = "model.pkl"

        # Upload model file to run outputs for history
        os.makedirs("outputs", exist_ok=True)
        output_path = os.path.join("outputs", model_file_name)
        joblib.dump(value=model, filename=output_path)

        # Upload model to parent run
        run.parent.upload_file(name=model_file_name, path_or_stream=output_path)

        run.complete()

    except Exception:
        exception = f"Exception: train_pipeline.py\n{traceback.format_exc()}"
        logger.error(exception)
        print(exception)
        exit(1)


if __name__ == "__main__":
    main()
