import os
import traceback
from argparse import ArgumentParser

import numpy as np
from azureml.core import Dataset, Workspace
from azureml.data.datapath import DataPath

from src.utils.pipelines import run_pipeline

workspace = None
datastore = None


def parse_args():
    # Parse command line arguments
    ap = ArgumentParser("score_pipeline_test")

    ap.add_argument("--subscription_id", required=True)
    ap.add_argument("--resource_group", required=True)
    ap.add_argument("--workspace_name", required=True)
    ap.add_argument("--pipeline_name", required=True)
    ap.add_argument("--dataset_name", required=True)
    ap.add_argument("--build_id", required=True)

    args, _ = ap.parse_known_args()
    return args


def copy_data_for_tests(dataset_name, file_path):
    # Define number of splits for test data
    n_splits = 5

    # Retreive dummy data from dataset and create splits
    dataset = Dataset.get_by_name(workspace, name=dataset_name)
    dataset_df = dataset.to_pandas_dataframe()
    dataset_df_splits = np.array_split(dataset_df.head(100), n_splits)

    os.makedirs("data", exist_ok=True)

    # Save data splits locally to upload to test datastore
    for idx, split_df in enumerate(dataset_df_splits):
        split_df.to_csv(f"data/split_{idx}.csv", index=False)

    # upload data to datastore for testing
    datastore.upload(src_dir="data", target_path=file_path)


def get_dataset_file(file_path):
    # Create file dataset from file path on test datastore
    dataset = Dataset.File.from_files(path=[(datastore, file_path)])

    # Find all files in the file dataset
    with dataset.mount() as mount_context:
        dataset_files = os.listdir(mount_context.mount_point)

    return dataset_files


def main():
    try:
        global workspace
        global datastore

        # Parse command line arguments
        args = parse_args()

        # Retreive workspace
        workspace = Workspace.get(
            subscription_id=args.subscription_id,
            resource_group=args.resource_group,
            name=args.workspace_name,
        )

        # Retreive default datastore for testing
        datastore = workspace.get_default_datastore()

        # Define directories for input and output test data on datastore
        input_file_path = f"tests/inputs/{args.build_id}"
        output_file_path = f"tests/outputs/{args.build_id}"

        print("Variable [input_file_path]:", input_file_path)
        print("Variable [output_file_path]:", output_file_path)

        # Copy data to input directory on datastore for testing
        copy_data_for_tests(args.dataset_name, input_file_path)

        # Define pipeline parameters
        pipeline_parameters = {
            "build_id": args.build_id,
            "input_datapath": DataPath(
                datastore=datastore, path_on_datastore=input_file_path
            ),
            "output_datapath": DataPath(
                datastore=datastore, path_on_datastore=output_file_path
            ),
        }

        print("Variable [pipeline_parameters]:", pipeline_parameters)

        # Run pipeline
        run_pipeline(workspace, args.pipeline_name, pipeline_parameters)

        # List all files in input and output datasets
        input_dataset_files = get_dataset_file(input_file_path)
        output_dataset_files = get_dataset_file(output_file_path)

        print("Variable [input_dataset_files]:", input_dataset_files)
        print("Variable [output_dataset_files]:", output_dataset_files)

        # Should have scored all input files and saved result to output datastore
        assert len(input_dataset_files) == len(output_dataset_files)

    except Exception:
        print(f"Exception: run_pipeline.py\n{traceback.format_exc()}")
        exit(1)


if __name__ == "__main__":
    main()
