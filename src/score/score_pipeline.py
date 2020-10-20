import sys
import traceback
from argparse import ArgumentParser

from azureml.core import Datastore, Environment, Workspace
from azureml.core.runconfig import RunConfiguration
from azureml.data.datapath import DataPath, DataPathComputeBinding
from azureml.pipeline.core import Pipeline, PipelineParameter
from azureml.pipeline.steps import PythonScriptStep

from src.utils.pipelines import draft_pipeline, publish_pipeline, run_pipeline

args = None


def parse_args(argv):
    # Parse command line arguments
    ap = ArgumentParser("score_pipeline")
    print(sys.argv[1:])

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
        ap.add_argument("--build_id", required=True)
        ap.add_argument("--input_datastore_name", required=True)
        ap.add_argument("--output_datastore_name", required=True)
        ap.add_argument("--environment_specification", required=True)
        ap.add_argument("--ai_connection_string", default="")
        ap.add_argument("--environment_name", default="train_env")
        ap.add_argument("--pipeline_metadata_file")

    if args.pipeline_action == "run":
        ap.add_argument("--input_datastore_name", required=True)
        ap.add_argument("--output_datastore_name", required=True)
        ap.add_argument("--input_datastore_path", required=True)
        ap.add_argument("--output_datastore_path", required=True)

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

    # Retreive input and output datastores
    input_datastore = Datastore(workspace, args.input_datastore_name)
    output_datastore = Datastore(workspace, args.output_datastore_name)

    # Define build id parameter
    build_id_param = PipelineParameter("build_id", default_value=args.build_id)

    # Define input datapath parameter
    input_datapath = DataPath(datastore=input_datastore, path_on_datastore="")
    input_datapath_param = (
        PipelineParameter(name="input_datapath", default_value=input_datapath),
        DataPathComputeBinding(mode="mount"),
    )

    # Define output datapath parameter
    output_datapath = DataPath(datastore=output_datastore, path_on_datastore="")
    output_datapath_param = (
        PipelineParameter(name="output_datapath", default_value=output_datapath),
        DataPathComputeBinding(mode="mount"),
    )

    # Define score step for pipeline
    score_step = PythonScriptStep(
        name="score_data",
        compute_target=compute_target,
        source_directory="src/score",
        script_name="score.py",
        inputs=[input_datapath_param, output_datapath_param],
        runconfig=run_config,
        allow_reuse=False,
        arguments=[
            "--build_id",
            build_id_param,
            "--input_datapath",
            input_datapath_param,
            "--output_datapath",
            output_datapath_param,
        ],
    )

    # Define pipeline for batch scoring
    pipeline = Pipeline(workspace=workspace, steps=[score_step])

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
            # Define pipeline parameters
            pipeline_parameters = {
                "build_id": args.build_id,
                "input_datapath": DataPath(
                    datastore=args.input_datastore_name,
                    path_on_datastore=args.input_datastore_path,
                ),
                "output_datapath": DataPath(
                    datastore=args.output_datastore_name,
                    path_on_datastore=args.output_datastore_path,
                ),
            }

            run_pipeline(workspace, args.pipeline_name, pipeline_parameters)

        elif args.pipeline_action == "publish":
            publish_pipeline(
                workspace, args.pipeline_name, args.disable_published_pipelines
            )

        else:
            raise Exception("Invalid pipeline action:", args.pipeline_action)

    except Exception:
        exception = f"Exception: train_pipeline.py\n{traceback.format_exc()}"
        print(exception)
        exit(1)


if __name__ == "__main__":
    main()
