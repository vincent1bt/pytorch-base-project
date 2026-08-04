import logging
import os
import argparse

logging.getLogger().setLevel(logging.INFO)
logging.info("Main file Started")

commit_sha = os.getenv('GIT_SHA') # raise error instead

parser = argparse.ArgumentParser()

parser.add_argument('--container_commit_sha', type=str)

args = parser.parse_args()

logging.getLogger().setLevel(logging.INFO)

logging.info(f"COMMIT SHA: {commit_sha}")

logging.info(f"Output Dir: {args.container_commit_sha}")

os.makedirs(os.path.dirname(args.container_commit_sha), exist_ok=True)

with open(args.container_commit_sha, "w") as f:
    f.write(commit_sha)

logging.info("Drift detected", extra={
    "json_fields": {
        "tag": "<DRIFT>",
        "message": "Drift detected on input feature 'age'",
        "error_type": "drift",
        "severity_detail": "input_distribution_shift",
        "context": {
	        "run_id": "kj882md",
            "batch_id": "batch_2026_07_02_001",
            "pipeline_run_id": "vertex-run-abc123",
            "model": "tps",
            "model_version": "3.2.1",
            "dataset": "2026/07",
        }
    }
})