from unittest.mock import MagicMock, patch

import numpy as np

from src.score.score import main, parse_args, score_data


def test_parse_args():
    mock_arguments = [
        "--build_id",
        "build_id_value",
        "--input_datapath",
        "input_datapath_value",
        "--output_datapath",
        "output_datapath_value",
    ]

    args = parse_args(mock_arguments)

    assert args.build_id is mock_arguments[1]
    assert args.input_datapath is mock_arguments[3]
    assert args.output_datapath is mock_arguments[5]


@patch("src.score.score.logger", MagicMock())
@patch("src.score.score.model")
@patch("src.score.score.pd")
def test_score_data(mock_df, mock_model, input_df):
    # Mock model predictions
    mock_model.predict_proba.return_value = np.array([[0.7, 0.3]] * input_df.shape[0])
    mock_df.read_csv.return_value = input_df

    # Run score data
    df = score_data("input")

    # Should include column for BMI
    assert "bmi" in df.columns.tolist()

    # Should include column for probabilities
    assert "probability" in df.columns.tolist()

    # Should include column for score
    assert "score" in df.columns.tolist()

    # Should include column for date
    assert "score_datetime" in df.columns.tolist()


@patch("src.score.score.Run", MagicMock())
@patch("src.score.score.logger", MagicMock())
@patch("src.score.score.datetime", MagicMock())
@patch("src.score.score.parse_args", MagicMock())
@patch("src.score.score.set_logger", MagicMock())
@patch("src.score.score.set_model", MagicMock())
@patch("src.score.score.os.chdir", MagicMock())
@patch("src.score.score.os.makedirs", MagicMock())
@patch("src.score.score.os.path.join", MagicMock())
@patch("src.score.score.score_data", MagicMock())
@patch("src.score.score.write_data")
@patch("src.score.score.glob")
def test_main(mock_glob, mock_write_data):
    # Mock file list to score
    mock_glob.glob.return_value = ["file_1", "file_2"]

    # Run main
    main()

    # Assert files have been passed to function to score
    mock_write_data.assert_called()
