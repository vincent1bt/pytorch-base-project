from google.cloud import aiplatform
import os

PROJECT_ID = os.getenv('PROJECT_ID')
LOCATION = os.getenv('REGION')
RUNNER_SERVICE_ACCOUNT = os.getenv('SERVICE_ACCOUNT')
GITHUB_SHA = os.getenv('GITHUB_SHA')

TEMPLATE_LOCATION = "https://us-central1-kfp.pkg.dev/{PROJECT_ID}/pipelines-repository/pytorch-test-pipeline/22bb7de88638"

aiplatform.init(project=PROJECT_ID, location=LOCATION)

job = aiplatform.PipelineJob(
    display_name="pytorch test pipeline run",
    template_path=TEMPLATE_LOCATION,
    parameter_values={
        "image_tag": GITHUB_SHA,
    },
)

job.submit(
    service_account=RUNNER_SERVICE_ACCOUNT
)