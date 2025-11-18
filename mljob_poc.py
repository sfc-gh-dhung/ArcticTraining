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

from snowflake.ml.jobs import submit_directory
from snowflake.snowpark import Session

session = Session.builder.create()

job = submit_directory(
    "arctic_training/launcher",
    entrypoint="ray_launcher.py",
    args=["causal/run-causal.yml"],
    stage_name="payload_stage",
    compute_pool="DHUNG_GPU_M",
    imports=["projects/causal/"],
    external_access_integrations=["allow_all_integration"],
    pip_requirements=["arctic-training"],
    target_instances=2,
    min_instances=1,
    session=session,
)
print(job.id)
print("Job completed with status: ", job.wait())
print(f"Job logs:\n\n{job.get_logs()}")
