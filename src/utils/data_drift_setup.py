import os
import sys
from argparse import ArgumentParser

from azureml.core import Dataset, Datastore, Workspace
from azureml.data.dataset_factory import DataType
from azureml.datadrift import DataDriftDetector

target_dataset_timestamp_column = "datetime"
input_schema_dir = os.path.join("input", "schema")
data_dir = "data"
input_schema_file = "schema.csv"


def parse_args(argv):
    ap = ArgumentParser("data_drift_setup")

    ap.add_argument("--subscription_id", required=True)
    ap.add_argument("--resource_group", required=True)
    ap.add_argument("--workspace_name", required=True)
    ap.add_argument("--target_dataset_path", required=True)
    ap.add_argument("--target_datastore_name", required=True)
    ap.add_argument("--baseline_dataset_name", required=True)
    ap.add_argument("--data_drift_monitor_name", required=True)
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--score_pipeline_endpoint_name", required=True)
    ap.add_argument("--compute_target", required=True)
    ap.add_argument("--feature_list", required=True)
    ap.add_argument("--frequency", default="Day")

    args, _ = ap.parse_known_args(argv)

    return args


def main():
    # Parse command line arguments
    args = parse_args(sys.argv[1:])

    # Retreive workspace
    workspace = Workspace.get(
        subscription_id=args.subscription_id,
        resource_group=args.resource_group,
        name=args.workspace_name,
    )

    # Retreive compute cluster
    compute_target = workspace.compute_targets[args.compute_target]

    # Get target and baseline datasets
    baseline_dataset = Dataset.get_by_name(workspace, args.baseline_dataset_name)

    # Retreive datastore for target dataset
    target_datastore = Datastore.get(workspace, args.target_datastore_name)

    # Upload sample data to the datastore
    # [Note: this step is required to ensure a data sample is present for validation when
    # registering a new target dataset below]
    os.makedirs(data_dir, exist_ok=True)
    baseline_dataset.take(1).to_pandas_dataframe().drop(
        ["cardiovascular_disease"], axis=1
    ).to_csv(os.path.join(data_dir, input_schema_file), index=False)
    target_datastore.upload(src_dir=data_dir, target_path=input_schema_dir)

    # Create a target dataset referencing the cloud location
    target_dataset = Dataset.Tabular.from_delimited_files(
        [(target_datastore, args.target_dataset_path)],
        validate=False,
        infer_column_types=False,
        set_column_types={
            "age": DataType.to_float(decimal_mark="."),
            "height": DataType.to_float(decimal_mark="."),
            "weight": DataType.to_float(decimal_mark="."),
            "systolic": DataType.to_float(decimal_mark="."),
            "diastolic": DataType.to_float(decimal_mark="."),
            "gender": DataType.to_string(),
            "cholesterol": DataType.to_string(),
            "glucose": DataType.to_string(),
            "smoker": DataType.to_string(),
            "alcoholic": DataType.to_string(),
            "active": DataType.to_string(),
            "datetime": DataType.to_datetime(),
        },
    )

    # Assign timestamp column for Tabular Dataset to activate time series related APIs
    target_dataset = target_dataset.with_timestamp_columns(
        timestamp=target_dataset_timestamp_column
    )

    # Get model id and version
    model_name, model_version = args.model_id.split(":")

    # Register dataset to Workspace
    target_dataset_name = f"{args.target_datastore_name}-{model_name}-{model_version}-{args.score_pipeline_endpoint_name}"

    target_dataset.register(
        workspace, target_dataset_name, create_new_version=True,
    )

    print("Variable [target_dataset]:", target_dataset)
    print("Variable [baseline_dataset]:", baseline_dataset)

    # Define features to monitor
    feature_list = args.feature_list.split(",")

    print("Variable [feature_list]:", args.feature_list)

    # List data drift detectors
    drift_detector_list = DataDriftDetector.list(workspace)

    # Delete existing data drift detector
    for drift_monitor in drift_detector_list:
        if drift_monitor.name == args.data_drift_monitor_name:
            print("Deleteing existing data drift monitor...")
            drift_monitor.delete()

    # Define data drift detector
    monitor = DataDriftDetector.create_from_datasets(
        workspace,
        args.data_drift_monitor_name,
        baseline_dataset,
        target_dataset,
        compute_target=compute_target,
        frequency=args.frequency,
        feature_list=feature_list,
    )

    print("Variable [monitor]:", monitor)

    # Enable the pipeline schedule for the data drift detector
    monitor.enable_schedule()


if __name__ == "__main__":
    main()
