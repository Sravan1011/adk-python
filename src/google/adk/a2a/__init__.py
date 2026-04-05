# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from a2a.types import Role
from a2a.types import TaskState


def _install_task_state_aliases() -> None:
  """Adds pre-1.0 TaskState aliases expected by ADK code and tests."""
  alias_by_name = {
      "working": "TASK_STATE_WORKING",
      "failed": "TASK_STATE_FAILED",
      "input_required": "TASK_STATE_INPUT_REQUIRED",
      "auth_required": "TASK_STATE_AUTH_REQUIRED",
      "completed": "TASK_STATE_COMPLETED",
      "submitted": "TASK_STATE_SUBMITTED",
      "canceled": "TASK_STATE_CANCELED",
      "unknown": "TASK_STATE_UNKNOWN",
  }
  for alias, canonical in alias_by_name.items():
    if not hasattr(TaskState, alias) and hasattr(TaskState, canonical):
      setattr(TaskState, alias, getattr(TaskState, canonical))


_install_task_state_aliases()


def _install_role_aliases() -> None:
  """Adds pre-1.0 Role aliases expected by ADK code and tests."""
  alias_by_name = {
      "user": "ROLE_USER",
      "agent": "ROLE_AGENT",
  }
  for alias, canonical in alias_by_name.items():
    if not hasattr(Role, alias) and hasattr(Role, canonical):
      setattr(Role, alias, getattr(Role, canonical))


_install_role_aliases()
