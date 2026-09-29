"""Optional interface for a pre-trained model on the largest selected dataset.

Use a suitable package directly, adapt this interface, or organise your own experiment.
"""

from __future__ import annotations

from typing import Any
from time import perf_counter
import numpy as np
from tabpfn import TabPFNClassifier
from sklearn.metrics import balanced_accuracy_score
from data_loading import DataSplits


def run_foundation_model(splits: DataSplits, seed: int) -> Any:
    """TODO: evaluate a pre-trained tabular foundation model.

    Choose and justify the model, how you use it, and the data available to it.
    Evaluate on the same complete test set as the forests, keeping it separate
    from training, adaptation, and model selection.
    Retain results for comparing performance and compute with the baseline
    and each tuned forest. See Section 3.5 of the assignment.
    """

    start_time = perf_counter()
    
    X_train_valid = np.concatenate([splits.X_train, splits.X_valid])
    y_train_valid = np.concatenate([splits.y_train, splits.y_valid])
    
    # Subsample to a viable in-context budget (e.g., 2000 rows)
    rng = np.random.default_rng(seed)
    max_context = min(2000, len(y_train_valid))
    indices = rng.choice(len(y_train_valid), size=max_context, replace=False)
    
    X_subset = np.asarray(X_train_valid[indices], dtype=np.float32)
    y_subset = np.asarray(y_train_valid[indices], dtype=np.int64)
    
    # Fit TabPFN (loads the context for inference)
    model = TabPFNClassifier(device='auto', random_state=seed)
    model.fit(X_subset, y_subset)
    
    # Evaluate on the exact same complete test set used by the forests
    X_test = np.asarray(splits.X_test, dtype=np.float32)
    y_pred = model.predict(X_test)
    
    metric = float(balanced_accuracy_score(splits.y_test, y_pred))
    
    return {
        "metrics": {"balanced_accuracy": metric},
        "elapsed_sec": float(perf_counter() - start_time),
        "context_samples": max_context
    }
