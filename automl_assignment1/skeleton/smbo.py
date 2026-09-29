"""Optional SMBO interface for the shared forest search space.

Implement the method, connect a suitable package, or replace this interface.
Choose and explain the method's settings and how you retain search results.
"""

from __future__ import annotations

from typing import Any
import optuna
from optuna.samplers import TPESampler
from random_forest import Config, Evaluator, SEARCH_SPACE


def optimise_smbo(
    evaluator: Evaluator,
    n_trials: int,
    n_trees: int,
    seed: int,
) -> tuple[Config, Any]:
    """TODO: use SMBO to choose configurations based on previous evaluations.

    Use the shared search space and train each forest with n_trees trees.
    Use up to n_trials evaluations, including any initial evaluations.
    Select the best configuration using the validation objective, respecting
    whether higher or lower values are better.
    Return the selected configuration and results needed for your analysis.
    """

    history = []

    def objective(trial: optuna.Trial) -> float:
        config = {
            "max_depth": trial.suggest_categorical("max_depth", SEARCH_SPACE["max_depth"]),
            "max_features": trial.suggest_categorical("max_features", SEARCH_SPACE["max_features"]),
            "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", SEARCH_SPACE["min_samples_leaf"]),
        }
        result = evaluator(config, n_trees, seed)
        history.append(result)
        return result["objective"]

    # Suppress verbose Optuna console logging to keep the output clean
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(sampler=sampler, direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return study.best_params, history
