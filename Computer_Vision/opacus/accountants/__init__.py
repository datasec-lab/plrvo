# Copyright (c) Meta Platforms, Inc. and affiliates.
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

from .accountant import IAccountant
from .gdp import GaussianAccountant
from .prv import PRVAccountant
from .rdp import RDPAccountant
from .rdp_PLRV import RDP_PLRVAccountant
from .laplace import LaplaceAccountant
from .lb import LBAccountant


__all__ = [
    "IAccountant",
    "GaussianAccountant",
    "RDPAccountant",
    "RDP_PLRVAccountant",
    "LaplaceAccountant",
    "LBAccountant",
]


def create_accountant(data_collector, mechanism: str) -> IAccountant:
    if mechanism == "rdp":
        return RDPAccountant(data_collector)
    elif mechanism == "rdp_plrv":
        return RDP_PLRVAccountant(data_collector)
    elif mechanism == "laplace":
        return LaplaceAccountant(data_collector)
    elif mechanism == "lb":
        return LBAccountant(data_collector)
    elif mechanism == "gdp":
        return GaussianAccountant()
    elif mechanism == "prv":
        return PRVAccountant()

    raise ValueError(f"Unexpected accounting mechanism: {mechanism}")
