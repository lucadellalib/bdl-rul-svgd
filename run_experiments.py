#!/usr/bin/env python3

# ==============================================================================
# Copyright 2022 Luca Della Libera and others.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Run the experiments."""

# Open a terminal and run:
# python run_experiments.py

from subprocess import call


__all__ = []


if __name__ == "__main__":
    for algorithm in ["bp", "bbb", "svgd"]:
        for model in ["d3", "c2p2"]:
            for subset in ["fd001", "fd002", "fd003", "fd004"]:
                call(["python", f"train_{algorithm}.py", model, subset])
