#!/usr/bin/env python3

import subprocess
import time
import logging

# Configuration
PARTITION = "debug"
JOB_NAME = "GC_Baseline"
SUBMIT_SCRIPT = "/iopsstor/scratch/cscs/stefschu/DSM500/github/experiments/02-Baseline/baseline.sbatch"
CHECK_INTERVAL = 60  # Time in seconds between checks
MAX_SUBMISSIONS = 24  # About 12h of runtime 

submission_count = 0

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # Print to console
)

def check_jobs():
    """Check if a job with the specified name is running in the debug partition."""
    try:
        result = subprocess.run(
            ["squeue", "--me", "--partition", PARTITION, "--format=%j", "--noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        job_names = result.stdout.strip().split("\n")
        job_names = [name.strip() for name in job_names if name.strip()]  # Remove empty lines
        return JOB_NAME in job_names  # True if job exists, False otherwise
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to check jobs: {e.stderr}")
        return True  # Assume jobs exist to avoid unnecessary submissions

def submit_job():
    """Submit a job using sbatch."""

    global submission_count

    if submission_count >= MAX_SUBMISSIONS:
        logging.warning(f"Max job submissions ({MAX_SUBMISSIONS}) reached. No more jobs will be submitted.")
        return

    try:
        result = subprocess.run(
            ["sbatch", SUBMIT_SCRIPT],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        submission_count += 1

        logging.info(f"Submitted job {submission_count}/{MAX_SUBMISSIONS}: {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Job submission failed: {e.stderr}")

def main():
    """Main loop for the watchdog."""
    global submission_count

    logging.info("SLURM Watchdog started.")
    
    while submission_count < MAX_SUBMISSIONS:
        if not check_jobs():
            logging.info(f"No '{JOB_NAME}' jobs found in the debug queue. Submitting a new job.")
            submit_job()
        else:
            logging.info(f"Job '{JOB_NAME}' found in the debug queue. No action needed.")
        
        time.sleep(CHECK_INTERVAL)
    
    logging.warning("Watchdog has reached the max submission limit and is now exiting.")

if __name__ == "__main__":
    main()
