# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from pathlib import Path
from typing import List

from snowflake.ml.jobs import MLJob
from snowflake.ml.jobs import submit_directory
from snowflake.snowpark import Session


def submit_job(
    config_file: str,
    compute_pool: str,
    stage_name: str,
    target_instances: int,
    min_instances: int,
    external_access_integrations: List[str],
    session: Session,
) -> MLJob:
    """Submit an Arctic Training job to Snowflake ML Jobs.

    Args:
        config_file: Path to the Arctic Training configuration file
        compute_pool: Name of the compute pool to use
        stage_name: Name of the stage for payload storage
        target_instances: Target number of instances
        min_instances: Minimum number of instances
        external_access_integrations: List of external access integrations
        session: Snowflake session object

    Raises:
        FileNotFoundError: If the config_file does not exist
    """
    # Convert to Path and validate existence
    config_file_obj = Path(config_file)
    if not config_file_obj.exists():
        raise FileNotFoundError(f"Arctic Training configuration file not found: {config_file}")

    # Derive relevant paths for the job submission
    config_parent = config_file_obj.parent
    config_remote_path = config_file_obj.relative_to(config_parent.parent)

    # FIXME: For this POC, we set the following special values:
    # - dir_path="arctic_training/launcher" -> this would be the directory containing the config file
    #                                          and any custom trainer code
    # - entrypoint: "ray_launcher.py"       -> this would be `arctic_training` in production
    # - imports: [config_file_obj.parent]   -> this would be unnecessary for production
    job = submit_directory(
        dir_path="arctic_training/launcher",
        entrypoint="ray_launcher.py",
        args=[config_remote_path.as_posix()],
        stage_name=stage_name,
        compute_pool=compute_pool,
        imports=[config_parent.as_posix()],
        external_access_integrations=external_access_integrations,
        pip_requirements=["arctic-training"],
        target_instances=target_instances,
        min_instances=min_instances,
        session=session,
    )
    return job


def main():
    parser = argparse.ArgumentParser(description="Submit an Arctic Training job to Snowflake ML Jobs")
    parser.add_argument(
        "config",
        type=str,
        help="Path to the Arctic Training configuration file",
    )
    parser.add_argument(
        "--compute-pool",
        type=str,
        default="DHUNG_GPU_M",
        help="Name of the compute pool to use (default: DHUNG_GPU_M)",
    )
    parser.add_argument(
        "--stage-name",
        type=str,
        default="payload_stage",
        help="Name of the stage for payload storage (default: payload_stage)",
    )
    parser.add_argument(
        "--target-instances",
        type=int,
        default=2,
        help="Target number of instances (default: 2)",
    )
    parser.add_argument(
        "--min-instances",
        type=int,
        default=1,
        help="Minimum number of instances (default: 1)",
    )
    parser.add_argument(
        "--external-access-integrations",
        type=str,
        nargs="+",
        default=["allow_all_integration"],
        help="List of external access integrations (default: allow_all_integration)",
    )
    args = parser.parse_args()

    session = Session.builder.create()
    job = submit_job(
        config_file=args.config,
        compute_pool=args.compute_pool,
        stage_name=args.stage_name,
        target_instances=args.target_instances,
        min_instances=args.min_instances,
        external_access_integrations=args.external_access_integrations,
        session=session,
    )
    print(job.id)
    print("Job completed with status: ", job.wait())
    print(f"Job logs:\n\n{job.get_logs()}")


if __name__ == "__main__":
    main()
