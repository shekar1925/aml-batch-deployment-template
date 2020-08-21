from unittest.mock import MagicMock, patch

from src.train.register import main, parse_args, register_model


def test_parse_args():
    mock_arguments = [
        "--model_name",
        "model_name_value",
        "--dataset_name",
        "dataset_name_value",
        "--build_id",
        "build_id_value",
    ]

    args = parse_args(mock_arguments)

    assert args.model_name is mock_arguments[1]
    assert args.dataset_name is mock_arguments[3]
    assert args.build_id is mock_arguments[5]


@patch("src.train.register.Dataset", MagicMock())
@patch("src.train.register.Workspace", MagicMock())
@patch("src.train.register.logger", MagicMock())
@patch("src.train.register.run")
def test_register_model(mock_run):
    # Mock run return values
    mock_run._run_id = "run_id_value"

    # Register model
    register_model("model_name", "dataset_name", "build_id")

    # Should have called run once to register model
    mock_run.parent.register_model.assert_called_once()


@patch("src.train.register.AzureLogHandler", MagicMock())
@patch("src.train.register.Run", MagicMock())
@patch("src.train.register.parse_args", MagicMock())
@patch("src.train.register.set_logger", MagicMock())
@patch("src.train.register.logger", MagicMock())
@patch("src.train.register.register_model")
def test_main(mock_register_model):
    # Execute main
    main()

    # Should have made a call to register model
    mock_register_model.assert_called_once()
