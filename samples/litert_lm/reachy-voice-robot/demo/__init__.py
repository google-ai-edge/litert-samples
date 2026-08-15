# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
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
# ==============================================================================

"""Voice robot demo on the Reachy Mini emulator.

The Mac captures video and displays the robot, the Pi does the compute. The
demo also runs standalone on the robot's own Pi, driving its camera, mic,
speaker, and motors directly (see demo/platform/reachy.py). The pipeline core
lives in emulator/; this package is just the I/O plumbing and HTTP bridge.
"""
