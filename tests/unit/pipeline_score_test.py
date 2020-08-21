from unittest.mock import MagicMock, patch

from src.score.score_pipeline import create_pipeline, main, parse_args


def test_parse_args_draft():
    mock_arguments = [
        "--subscription_id",
        "subscription_id_value",
        "--resource_group",
        "resource_group_value",
        "--workspace_name",
        "workspace_name_value",
        "--pipeline_name",
        "pipeline_name_value",
        "--compute_target",
        "compute_target_value",
        "--experiment_name",
        "experiment_name_value",
        "--build_id",
        "build_id_value",
        "--input_datastore_name",
        "input_datastore_name_value",
        "--output_datastore_name",
        "output_datastore_name_value",
        "--environment_specification",
        "environment_specification_value",
        "--ai_connection_string",
        "ai_connection_string_value",
        "--environment_name",
        "environment_name_value",
        "--pipeline_metadata_file",
        "pipeline_metadata_file_value",
        "--pipeline_action",
        "draft",
    ]

    args = parse_args(mock_arguments)

    assert args.subscription_id is mock_arguments[1]
    assert args.resource_group is mock_arguments[3]
    assert args.workspace_name is mock_arguments[5]
    assert args.pipeline_name is mock_arguments[7]
    assert args.compute_target is mock_arguments[9]
    assert args.experiment_name is mock_arguments[11]
    assert args.build_id is mock_arguments[13]
    assert args.input_datastore_name is mock_arguments[15]
    assert args.output_datastore_name is mock_arguments[17]
    assert args.environment_specification is mock_arguments[19]
    assert args.ai_connection_string is mock_arguments[21]
    assert args.environment_name is mock_arguments[23]
    assert args.pipeline_metadata_file is mock_arguments[25]


def test_parse_args_run():
    mock_arguments = [
        "--subscription_id",
        "subscription_id_value",
        "--resource_group",
        "resource_group_value",
        "--workspace_name",
        "workspace_name_value",
        "--pipeline_name",
        "pipeline_name_value",
        "--input_datastore_name",
        "input_datastore_name_value",
        "--output_datastore_name",
        "output_datastore_name_value",
        "--input_datastore_path",
        "input_datastore_path_value",
        "--output_datastore_path",
        "output_datastore_path_value",
        "--pipeline_action",
        "run",
    ]

    args = parse_args(mock_arguments)

    assert args.subscription_id is mock_arguments[1]
    assert args.resource_group is mock_arguments[3]
    assert args.workspace_name is mock_arguments[5]
    assert args.pipeline_name is mock_arguments[7]
    assert args.input_datastore_name is mock_arguments[9]
    assert args.output_datastore_name is mock_arguments[11]
    assert args.input_datastore_path is mock_arguments[13]
    assert args.output_datastore_path is mock_arguments[15]


def test_parse_args_publish():
    mock_arguments = [
        "--subscription_id",
        "subscription_id_value",
        "--resource_group",
        "resource_group_value",
        "--workspace_name",
        "workspace_name_value",
        "--pipeline_name",
        "pipeline_name_value",
        "--disable_published_pipelines",
        "--pipeline_action",
        "publish",
    ]

    args = parse_args(mock_arguments)

    assert args.subscription_id is mock_arguments[1]
    assert args.resource_group is mock_arguments[3]
    assert args.workspace_name is mock_arguments[5]
    assert args.pipeline_name is mock_arguments[7]
    assert args.disable_published_pipelines is True


@patch("src.score.score_pipeline.Datastore", MagicMock())
@patch("src.score.score_pipeline.DataPath", MagicMock())
@patch("src.score.score_pipeline.Environment", MagicMock())
@patch("src.score.score_pipeline.PipelineParameter", MagicMock())
@patch("src.score.score_pipeline.args", MagicMock())
@patch("src.score.score_pipeline.PythonScriptStep")
@patch("src.score.score_pipeline.Pipeline")
@patch("src.score.score_pipeline.Workspace")
def test_create_pipeline(
    mock_workspace, mock_pipeline_publish, mock_python_script_step,
):
    # Run main
    pipeline = create_pipeline(mock_workspace)

    # Should return a value
    assert pipeline is not None

    # Should make calls to python script step
    mock_python_script_step.assert_called()

    # Should make call to create pipeline
    mock_pipeline_publish.assert_called_once()


@patch("src.score.score_pipeline.Workspace", MagicMock())
@patch("src.score.score_pipeline.publish_pipeline")
@patch("src.score.score_pipeline.run_pipeline")
@patch("src.score.score_pipeline.draft_pipeline")
@patch("src.score.score_pipeline.create_pipeline")
@patch("src.score.score_pipeline.parse_args")
def test_create_pipeline_draft(
    mock_args,
    mock_create_pipeline,
    mock_draft_pipeline,
    mock_run_pipeline,
    mock_publish_pipeline,
):
    # Mock pipeline action attribute
    mock_args.return_value = MagicMock(pipeline_action="draft")

    # Run main
    main()

    # Should create pipline
    mock_create_pipeline.assert_called_once()

    # Should make call to draft a pipeline
    mock_draft_pipeline.assert_called_once()

    # Should not make call to run a pipeline
    mock_run_pipeline.assert_not_called()

    # Should not make call to publish a pipeline
    mock_publish_pipeline.assert_not_called()


@patch("src.score.score_pipeline.Workspace", MagicMock())
@patch("src.score.score_pipeline.publish_pipeline")
@patch("src.score.score_pipeline.run_pipeline")
@patch("src.score.score_pipeline.draft_pipeline")
@patch("src.score.score_pipeline.parse_args")
def test_create_pipeline_run(
    mock_args, mock_draft_pipeline, mock_run_pipeline, mock_publish_pipeline,
):
    # Mock pipeline action attribute
    mock_args.return_value = MagicMock(pipeline_action="run")

    # Run main
    main()

    # Should make call to draft a pipeline
    mock_draft_pipeline.assert_not_called()

    # Should not make call to run a pipeline
    mock_run_pipeline.assert_called_once()

    # Should not make call to publish a pipeline
    mock_publish_pipeline.assert_not_called()


@patch("src.score.score_pipeline.Workspace", MagicMock())
@patch("src.score.score_pipeline.publish_pipeline")
@patch("src.score.score_pipeline.run_pipeline")
@patch("src.score.score_pipeline.draft_pipeline")
@patch("src.score.score_pipeline.parse_args")
def test_create_pipeline_publish(
    mock_args, mock_draft_pipeline, mock_run_pipeline, mock_publish_pipeline,
):
    # Mock pipeline action attribute
    mock_args.return_value = MagicMock(pipeline_action="publish")

    # Run main
    main()

    # Should make call to draft a pipeline
    mock_draft_pipeline.assert_not_called()

    # Should not make call to run a pipeline
    mock_run_pipeline.assert_not_called()

    # Should not make call to publish a pipeline
    mock_publish_pipeline.assert_called_once()
