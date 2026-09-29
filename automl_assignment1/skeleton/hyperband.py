"""Optional Hyperband interface for random forests.

Implement the method, connect a suitable package, or replace this interface.
Choose and justify the allocation schedule and how you retain search results.
"""

from __future__ import annotations

from typing import Any
import math
import numpy as np
from random_forest import Config, Evaluator, sample_configuration


def optimise_hyperband(
    evaluator: Evaluator,
    min_trees: int,
    max_trees: int,
    seed: int,
    reduction_factor: int = 3,
) -> tuple[Config, Any]:
    """TODO: implement or configure multiple successive-halving brackets.

    Use the shared search space. Start brackets with different numbers of
    configurations and trees per forest, between min_trees and max_trees.
    At each stage, keep the better configurations and give them more trees,
    keeping other settings fixed. Use reduction_factor for the decrease in
    configuration count and increase in trees.
    Compare validation objectives consistently: respect whether higher or lower
    values are better. Explain your schedule, refitting or warm starts, and
    how validation results determine the final selection.
    Return the selected configuration and results needed for your analysis.
    """

    rng = np.random.default_rng(seed)
    s_max = int(math.floor(math.log(max_trees / min_trees, reduction_factor)))
    B = (s_max + 1) * max_trees
    
    history = []
    best_config = {}
    best_objective = -float("inf")
    
    for s in reversed(range(s_max + 1)):
        # Initial configurations for this bracket
        n = int(math.ceil(int(B / max_trees / (s + 1)) * (reduction_factor ** s)))
        r = max_trees * (reduction_factor ** (-s))
        configs = [sample_configuration(rng) for _ in range(n)]
        
        # Successive halving within the bracket
        for i in range(s + 1):
            n_i = int(n * (reduction_factor ** (-i)))
            r_i = max(min_trees, int(r * (reduction_factor ** i))) 
            
            stage_results = []
            for config in configs:
                res = evaluator(config, r_i, seed)
                history.append(res)
                stage_results.append((res["objective"], config))
                
                if res["objective"] > best_objective:
                    best_objective = res["objective"]
                    best_config = config
            
            # Sort descending since higher Balanced Accuracy is better
            stage_results.sort(key=lambda x: x[0], reverse=True)
            
            # Keep top 1/eta configurations
            num_to_keep = max(1, int(n_i / reduction_factor))
            configs = [c for _, c in stage_results[:num_to_keep]]
            
    return best_config, history
