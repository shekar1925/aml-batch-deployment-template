import glob
import logging
import os
import sys
import traceback
from argparse import ArgumentParser
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from azureml.core import Run
from azureml.core.model import Model
from opencensus.ext.azure.log_exporter import AzureLogHandler

run = None
model = None
logger = None
file_type = "*.csv"


def parse_args(argv):
    ap = ArgumentParser("score")

    ap.add_argument("--build_id", required=True)
    ap.add_argument("--input_datapath", required=True)
    ap.add_argument("--output_datapath", required=True)

    args, _ = ap.parse_known_args(argv)

    return args


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


def set_model(build_id):
    global model

    # Retreive workspace
    workspace = run.experiment.workspace

    # Find models with build id
    model_list = Model.list(workspace, tags=[["build_id", build_id]], latest=True)

    # Throw error if no model is found
    if not model_list:
        raise Exception(f"Model not found: build_id={build_id}")

    # Retreive path to model folder
    model_path = Model.get_model_path(model_list[0].name, version=model_list[0].version)

    # Deserialize the model file back into a sklearn model
    model = joblib.load(model_path)

    print("Retreived model:", {"model_id": model_list[0].id})


def score_data(input_file_path):
    # Read file
    df = pd.read_csv(input_file_path)

    # Convert strings to float
    df = df.astype(
        {
            "age": np.float64,
            "height": np.float64,
            "weight": np.float64,
            "systolic": np.float64,
            "diastolic": np.float64,
        }
    )

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
    raw_numeric_features = ["age", "systolic", "diastolic", "height", "weight"]

    # Filter dataframe for columns of interest
    df = df[categorical_features + raw_numeric_features]

    # Create feature for Body Mass Index (indicator of heart health)
    df["bmi"] = df.weight / (df.height / 100) ** 2

    # Get model features / target
    df = df.drop(labels=["height", "weight"], axis=1)

    # Convert data types of model features
    df[categorical_features] = df[categorical_features].astype(np.object)
    df[numeric_features] = df[numeric_features].astype(np.float64)

    # Preprocess payload and get model prediction
    probability = model.predict_proba(df)

    # Add prediction, confidence level and datetime to input data as columns
    df["probability"] = probability[:, 1]
    df["score"] = np.where(probability[:, 1] >= 0.5, 1, 0)
    df["score_datetime"] = datetime.now()

    return df


def write_data(df, output_file_path):
    # Write scored results
    df.to_csv(output_file_path, index=False)
    print("Completed File:", output_file_path)
    logger.info({"output_file_path": output_file_path})


def main():
    try:
        global run

        # Retrieve current service context
        run = Run.get_context()

        # Parse command line arguments
        args = parse_args(sys.argv[1:])

        # Print argument values
        print("Argument [build_id]:", args.build_id)
        print("Argument [input_datapath]:", args.input_datapath)
        print("Argument [output_datapath]:", args.output_datapath)

        # Initialise model and logger
        set_model(args.build_id)
        set_logger()

        # Change current working directory to input_datapath
        os.chdir(args.input_datapath)

        # Create output directory
        os.makedirs(args.output_datapath, exist_ok=True)

        # Define files to score
        files_to_score = glob.glob(file_type)
        print("Scoring files:", files_to_score)
        logger.info({"files_to_score": files_to_score})

        # Score files
        for idx, file_name in enumerate(files_to_score):
            # Define file path for data
            input_file_path = os.path.join(args.input_datapath, file_name)

            # Define file path to write results
            current_date = datetime.today().strftime("%Y_%m_%d_%H_%M")
            output_file_name = f"{current_date}_{idx}.csv"
            output_file_path = os.path.join(args.output_datapath, output_file_name)

            # Score file and write results to output directory
            df = score_data(input_file_path)
            write_data(df, output_file_path)

        print("Completed Job")

    except Exception:
        exception = f"Exception: train_pipeline.py\n{traceback.format_exc()}"
        logger.error(exception)
        print(exception)
        exit(1)


if __name__ == "__main__":
    main()
