import json

from azureml.pipeline.core import PipelineDraft
from azureml.pipeline.core.graph import PublishedPipeline


def write_pipeline_metadata(pipeline, metadata_file):
    # Get pipeline details
    pipeline_metadata = {
        "id": pipeline.id,
        "name": pipeline.name,
    }

    # Write pipeline details to file
    if metadata_file:
        with open(metadata_file, "w") as f:
            json.dump(pipeline_metadata, f)

    return pipeline_metadata


def get_pipeline_draft(workspace, pipeline_name):
    # List all draft pipelines on workspace
    draft_pipelines = PipelineDraft.list(workspace)

    # Retreive draft pipeline by name if it exists
    for pipeline_draft in draft_pipelines:
        if pipeline_draft.name == pipeline_name:
            print(
                "Retreived existing pipeline draft:",
                {"id": pipeline_draft.id, "name": pipeline_draft.name},
            )
            return pipeline_draft

    return None


def create_pipeline_draft(
    workspace, pipeline, pipeline_name, experiment_name, build_id="--"
):
    # Get existing pipelines draft
    existing_pipeline = get_pipeline_draft(workspace, pipeline_name)

    print("get_pipeline_draft_test", existing_pipeline)

    # Delete existing pipeline draft if exists
    if existing_pipeline:
        existing_pipeline.delete()
        print(
            "Deleted existing pipeline draft:",
            {"id": existing_pipeline.id, "name": existing_pipeline.name},
        )

    # Create new pipeline if draft does not exsit
    pipeline_tags = {"build_id": build_id}
    pipeline_draft = PipelineDraft.create(
        workspace=workspace,
        pipeline=pipeline,
        name=pipeline_name,
        experiment_name=experiment_name,
        tags=pipeline_tags,
    )

    return pipeline_draft


def disable_pipeline(pipeline):
    try:
        pipeline.disable()
        print(
            "Disabled existing published pipeline:",
            {"id": pipeline.id, "name": pipeline.name},
        )
    except Exception:
        print(
            "Unable to disabled existing published pipeline:",
            {"id": pipeline.id, "name": pipeline.name},
        )


def disable_existing_published_pipelines(workspace, pipeline_name):
    # List all draft pipelines on workspace
    published_pipelines = PublishedPipeline.list(workspace)

    # Retreive draft pipeline by name if it exists
    for pipeline_published in published_pipelines:
        if pipeline_published.name == pipeline_name:
            disable_pipeline(pipeline_published)


def draft_pipeline(
    workspace, pipeline, pipeline_name, experiment_name, build_id, metadata_file
):
    # Create new pipeline
    pipeline = create_pipeline_draft(
        workspace, pipeline, pipeline_name, experiment_name, build_id
    )

    # Log pipeline draft metadata
    pipeline_metadata = write_pipeline_metadata(pipeline, metadata_file)
    print("Created new pipeline draft:", pipeline_metadata)


def run_pipeline(workspace, pipeline_name, pipeline_parameters=None):
    # Trigger pipeline run and wait for completion
    pipeline = get_pipeline_draft(workspace, pipeline_name)

    # Update pipeline parameters
    if pipeline_parameters:
        pipeline.update(pipeline_parameters=pipeline_parameters)

    # Run pipeline draft
    pipeline_run = pipeline.submit_run()
    pipeline_run.wait_for_completion()


def publish_pipeline(workspace, pipeline_name, disable_published_pipelines=False):
    # Get pipeline draft
    pipeline = get_pipeline_draft(workspace, pipeline_name)

    # Disable existing published pipelines
    if disable_published_pipelines:
        disable_existing_published_pipelines(workspace, pipeline_name)

    # Publish pipeline and remove draft
    pipeline.publish()
    pipeline.delete()
    print("Pipeline published")
