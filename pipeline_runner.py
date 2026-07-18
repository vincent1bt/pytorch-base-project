from google.cloud import aiplatform
import os
import logging

PROJECT_ID = os.getenv('PROJECT_ID')
LOCATION = os.getenv('REGION')
RUNNER_SERVICE_ACCOUNT = os.getenv('SERVICE_ACCOUNT')
GITHUB_SHA = os.getenv('GITHUB_SHA')

TEMPLATE_LOCATION = f"https://us-central1-kfp.pkg.dev/{PROJECT_ID}/pipelines-repository/pytorch-test-pipeline/latest"

aiplatform.init(project=PROJECT_ID, location=LOCATION)

logging.getLogger().setLevel(logging.INFO)

logging.info(f"TEMPLATE USED: {TEMPLATE_LOCATION}")
logging.info(f"SHA USED: {GITHUB_SHA}")

job = aiplatform.PipelineJob(
    display_name="pytorch test pipeline run",
    template_path=TEMPLATE_LOCATION,
    parameter_values={
        "commit_hash": GITHUB_SHA,
    },
)

job.submit(
    service_account=RUNNER_SERVICE_ACCOUNT
)