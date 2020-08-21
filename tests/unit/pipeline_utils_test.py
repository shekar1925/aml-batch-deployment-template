from unittest.mock import MagicMock, patch

from src.utils.pipelines import draft_pipeline, publish_pipeline, run_pipeline


@patch("azureml.pipeline.core.PipelineDraft.create", MagicMock())
@patch("json.dump")
@patch("azureml.pipeline.core.PipelineDraft.list")
def test_draft_pipeline(mock_pipeline_draft_list, mock_json_dump):
    # Mock pipeline draft return list
    draft_one = MagicMock(id="pipeline_id_value")
    draft_one.name = "pipeline_name_value"
    mock_pipeline_draft_list.return_value = [draft_one]

    # Run pipeline draft
    draft_pipeline(
        "workspace_value",
        "pipeline_value",
        "pipeline_name_value",
        "experiment_name_value",
        "build_id_value",
        "metadata_file_value",
    )

    # Should write metadata to file
    mock_json_dump.assert_called()


@patch("azureml.pipeline.core.PipelineDraft.list")
def test_run_pipeline(mock_pipeline_draft):
    # Mock existing pipline draft
    existing_pipeline = MagicMock(id="pipeline_id_value")
    existing_pipeline.name = "pipeline_name_value"

    # Mock pipeline run object
    mock_pipeline_run = MagicMock()
    existing_pipeline.submit_run.return_value = mock_pipeline_run

    # Mock pipeline draft list
    mock_pipeline_draft.return_value = [existing_pipeline]

    # Run pipeline draft
    run_pipeline(
        "workspace_value", "pipeline_name_value",
    )

    # Should submit pipeline run
    existing_pipeline.submit_run.assert_called()

    # Should wait for pipeline run to complete
    mock_pipeline_run.wait_for_completion.assert_called()


@patch("azureml.pipeline.core.PublishedPipeline.list")
@patch("azureml.pipeline.core.PipelineDraft.list")
def test_publish_pipeline(mock_pipeline_draft, mock_pipeline_published):
    # Mock existing published pipline
    published_pipeline = MagicMock(id="published_pipeline_id_value")
    published_pipeline.name = "pipeline_name_value"

    # Mock existing draft pipline
    draft_pipeline = MagicMock(id="draft_pipeline_id_value")
    draft_pipeline.name = "pipeline_name_value"

    # Mock pipeline lists
    mock_pipeline_draft.return_value = [draft_pipeline]
    mock_pipeline_published.return_value = [published_pipeline]

    publish_pipeline(
        "workspace_value", "pipeline_name_value", disable_published_pipelines=False
    )
    # Should publish pipeline draft
    draft_pipeline.publish.assert_called()

    # Should delete pipeline draft
    draft_pipeline.delete.assert_called()
