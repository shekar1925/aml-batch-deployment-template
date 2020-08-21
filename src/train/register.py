import logging
import sys
import traceback
from argparse import ArgumentParser

import sklearn
from azureml.core import Dataset, Run, Workspace
from azureml.core.model import Model
from opencensus.ext.azure.log_exporter import AzureLogHandler

run = None
logger = None
evaluation_metric = "test_acccuracy"
evaluation_metric_threshold = 0.7


def set_logger():
    global run
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


def parse_args(argv):
    ap = ArgumentParser("register")
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--dataset_name", required=True)
    ap.add_argument("--build_id", required=True)

    args, _ = ap.parse_known_args(argv)
    return args


def register_model(model_name, dataset_name, build_id):
    # Retreive dataset
    if run._run_id.startswith("OfflineRun"):
        workspace = Workspace.from_config()
    else:
        workspace = run.experiment.workspace

    # Retreive train datasets
    train_dataset = [(dataset_name, Dataset.get_by_name(workspace, name=dataset_name))]

    # Get evaluation metric for model
    run_metrics = run.parent.get_metrics()

    # Define model file name
    model_file_name = "model.pkl"

    # Define model tags
    model_tags = {
        "build_id": build_id,
        "test_acccuracy": run_metrics.get(evaluation_metric),
    }

    print("Variable [model_tags]:", model_tags)

    # Register the model
    model = run.parent.register_model(
        model_name=model_name,
        model_path=model_file_name,
        model_framework=Model.Framework.SCIKITLEARN,
        model_framework_version=sklearn.__version__,
        datasets=train_dataset,
        tags=model_tags,
    )

    print("Variable [model]:", model.serialize())
    logger.info(model.serialize())


def main():
    try:
        global run

        # Retrieve current service context
        run = Run.get_context()

        # Set logger
        set_logger()

        # Parse command line arguments
        args = parse_args(sys.argv[1:])

        # Print argument values
        print("Argument [model_name]:", args.model_name)
        print("Argument [dataset_name]:", args.dataset_name)
        print("Argument [build_id]:", args.build_id)

        # Get evaluation metric for model
        run_metrics = run.parent.get_metrics()

        model_metric = float(run_metrics.get(evaluation_metric))
        print("Variable [model_metric]:", model_metric)

        # Register model if performance is better than threshold or cancel run
        if model_metric > evaluation_metric_threshold:
            register_model(
                args.model_name, args.dataset_name, args.build_id,
            )
        else:
            run.parent.cancel()

    except Exception:
        exception = f"Exception: train_pipeline.py\n{traceback.format_exc()}"
        logger.error(exception)
        print(exception)
        exit(1)


if __name__ == "__main__":
    main()
