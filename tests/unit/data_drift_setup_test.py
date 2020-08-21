from unittest.mock import MagicMock, patch

from src.utils.data_drift_setup import main, parse_args


def test_parse_args_draft():
    mock_arguments = [
        "--subscription_id",
        "subscription_id_value",
        "--resource_group",
        "resource_group_value",
        "--workspace_name",
        "workspace_name_value",
        "--target_dataset_path",
        "target_dataset_path_value",
        "--target_datastore_name",
        "target_datastore_name_value",
        "--baseline_dataset_name",
        "baseline_dataset_value",
        "--data_drift_monitor_name",
        "data_drift_monitor_name_value",
        "--model_id",
        "model_id_value",
        "--score_pipeline_endpoint_name",
        "score_pipeline_endpoint_name_value",
        "--compute_target",
        "compute_target_value",
        "--feature_list",
        "feature_list_value",
        "--frequency",
        "frequency_value",
    ]

    args = parse_args(mock_arguments)

    assert args.subscription_id == mock_arguments[1]
    assert args.resource_group == mock_arguments[3]
    assert args.workspace_name == mock_arguments[5]
    assert args.target_dataset_path == mock_arguments[7]
    assert args.target_datastore_name == mock_arguments[9]
    assert args.baseline_dataset_name == mock_arguments[11]
    assert args.data_drift_monitor_name == mock_arguments[13]
    assert args.model_id == mock_arguments[15]
    assert args.score_pipeline_endpoint_name == mock_arguments[17]
    assert args.compute_target == mock_arguments[19]
    assert args.feature_list == mock_arguments[21]
    assert args.frequency == mock_arguments[23]


@patch("src.utils.data_drift_setup.Dataset", MagicMock())
@patch("src.utils.data_drift_setup.Datastore", MagicMock())
@patch("src.utils.data_drift_setup.DataType", MagicMock())
@patch("src.utils.data_drift_setup.Workspace", MagicMock())
@patch("src.utils.data_drift_setup.os", MagicMock())
@patch("src.utils.data_drift_setup.parse_args")
@patch("src.utils.data_drift_setup.DataDriftDetector")
def test_data_drift_setup(mock_data_drift_detector, mock_parse_args):
    # Mock data drift monitor and arguments
    mock_monitor = MagicMock()
    mock_data_drift_detector.create_from_datasets.return_value = mock_monitor
    mock_parse_args.return_value = MagicMock(model_id="model_name:model_version")

    # Run main
    main()

    # Should enable data drift monitor
    mock_monitor.enable_schedule.assert_called_once()
