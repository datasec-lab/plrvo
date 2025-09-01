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

from typing import List, Optional, Tuple, Union
import numpy as np
from .accountant import IAccountant
from .analysis import lb as privacy_analysis
from .analysis import rdp as gaussian_analysis


class LBAccountant(IAccountant):
    DEFAULT_ALPHAS = range(2,203)

    def __init__(self, data_collector):
        super().__init__(data_collector)
        self.args = {}

    def step(self, noise_multiplier, sample_rate):
        self.sample_rate = sample_rate
        if len(self.history) >= 1:
            last_args, num_steps = self.history.pop()
            if (
                last_args == self.args
            ):
                self.history.append(
                    (last_args, num_steps + 1)
                )
            else:
                self.history.append(
                    (last_args, num_steps)
                )
                self.history.append((self.args, 1))

        else:
            self.history.append((self.args, 1))
            
    def get_privacy_spent(
        self, *, delta: float, alphas: Optional[List[Union[float, int]]] = None
    ) -> Tuple[float, float]:
        if not self.history:
            return 0, 0
        
        best_alpha = 0
        epsis = []
        
        if alphas is None:
            alphas = self.DEFAULT_ALPHAS
        rdp = sum(
            [
                privacy_analysis.compute_rdp_subsample(
                    args = args,
                    num_steps=num_steps,
                    delta=delta,
                    orders=alphas,
                    sample_rate=self.sample_rate,
                )
                for (args, num_steps) in self.history
            ]
        )
        #return(rdp[0], alphas[0])
        eps, best_alpha = privacy_analysis.get_privacy_spent(
            orders=alphas, rdp=rdp, delta=delta
        )
        
        return float(eps), float(best_alpha)

    def get_epsilon(
        self, delta: float, alphas: Optional[List[Union[float, int]]] = None
    ):
        """
        Return privacy budget (epsilon) expended so far.

        Args:
            delta: target delta
            alphas: List of RDP orders (alphas) used to search for the optimal conversion
                between RDP and (epd, delta)-DP
        """
        eps, _ = self.get_privacy_spent(delta=delta, alphas=alphas)
        return eps

    def __len__(self):
        return len(self.history)

    @classmethod
    def mechanism(cls) -> str:
        return "rdp_plrv"
