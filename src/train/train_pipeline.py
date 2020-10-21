import sys
import traceback
from argparse import ArgumentParser

from azureml.core import Environment, Workspace
from azureml.core.dataset import Dataset
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import Pipeline
from azureml.pipeline.core.graph import PipelineParameter
from azureml.pipeline.steps import PythonScriptStep
from src.utils.pipelines import draft_pipeline, publish_pipeline, run_pipeline

args = None


def parse_args(argv):
    # Parse command line arguments
    ap = ArgumentParser("train_pipeline")
    print(sys.argv[1:])

    # common arguments
    ap.add_argument("--subscription_id", required=True)
    ap.add_argument("--resource_group", required=True)
    ap.add_argument("--workspace_name", required=True)
    ap.add_argument("--pipeline_name", required=True)
    ap.add_argument(
        "--pipeline_action", choices=["draft", "run", "publish"], required=True
    )

    args, _ = ap.parse_known_args(argv)
    print(args)

    # check draft arguments are present
    if args.pipeline_action == "draft":
        ap.add_argument("--compute_target", required=True)
        ap.add_argument("--experiment_name", required=True)
        ap.add_argument("--dataset_name", required=True)
        ap.add_argument("--model_name", required=True)
        ap.add_argument("--build_id", required=True)
        ap.add_argument("--environment_specification", required=True)
        ap.add_argument("--ai_connection_string", default="")
        ap.add_argument("--environment_name", default="train_env")
        ap.add_argument("--pipeline_metadata_file", required=True)

        args, _ = ap.parse_known_args(argv)

    # check publish arguments are present
    if args.pipeline_action == "publish":
        ap.add_argument("--disable_published_pipelines", action="store_true")

    args, _ = ap.parse_known_args(argv)
    return args


def create_pipeline(workspace):
    # Retreive compute cluster
    compute_target = workspace.compute_targets[args.compute_target]

    # Setup batch scoring environment from conda dependencies
    environment = Environment.from_conda_specification(
        name=args.environment_name, file_path=args.environment_specification
    )

    # Add environment variables
    environment.environment_variables = {
        "APPLICATIONINSIGHTS_CONNECTION_STRING": args.ai_connection_string
    }

    # Enable docker run
    environment.docker.enabled = True

    # Create run config
    run_config = RunConfiguration()
    run_config.environment = environment

    # Retreive input dataset
    input_dataset = Dataset.get_by_name(workspace, name=args.dataset_name)

    # Define model name paramater
    model_name_param = PipelineParameter(
        name="model_name", default_value=args.model_name
    )

    # Define dataset name paramater
    dataset_name_param = PipelineParameter(
        name="dataset_name", default_value=args.dataset_name
    )

    # Define build id paramater
    build_id_param = PipelineParameter(name="build_id", default_value=args.build_id)

    # Define train model step for pipeline
    train_step = PythonScriptStep(
        name="train_model",
        compute_target=compute_target,
        source_directory="src/train",
        script_name="train.py",
        inputs=[input_dataset.as_named_input("InputDataset")],
        runconfig=run_config,
        allow_reuse=False,
    )

    # Define register model step for pipeline
    register_step = PythonScriptStep(
        name="register_model",
        compute_target=compute_target,
        source_directory="src/train",
        script_name="register.py",
        runconfig=run_config,
        allow_reuse=False,
        arguments=[
            "--model_name",
            model_name_param,
            "--dataset_name",
            dataset_name_param,
            "--build_id",
            build_id_param,
        ],
    )

    # Define step order
    register_step.run_after(train_step)

    # Define pipeline for scoring
    pipeline = Pipeline(workspace=workspace, steps=[train_step, register_step])

    return pipeline


def main():
    try:
        global args

        # Parse command line arguments
        args = parse_args(sys.argv[1:])

        # Retreive workspace
        workspace = Workspace.get(
            subscription_id=args.subscription_id,
            resource_group=args.resource_group,
            name=args.workspace_name,
        )

        print("args, args.pipeline_action", args, args.pipeline_action)

        if args.pipeline_action == "draft":
            pipeline = create_pipeline(workspace)
            draft_pipeline(
                workspace,
                pipeline,
                args.pipeline_name,
                args.experiment_name,
                args.build_id,
                args.pipeline_metadata_file,
            )

        elif args.pipeline_action == "run":
            run_pipeline(workspace, args.pipeline_name)

        elif args.pipeline_action == "publish":
            publish_pipeline(
                workspace, args.pipeline_name, args.disable_published_pipelines
            )

        else:
            raise Exception("Invalid pipeline action:", args.pipeline_action)

    except Exception:
        print(f"Exception: train_pipeline.py\n{traceback.format_exc()}")
        exit(1)


if __name__ == "__main__":
    main()
