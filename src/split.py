"""
Subject Split Module for KFall Fall Detection.

Splits subjects into Train, Validation, and Test sets deterministically
to prevent data leakage across sliding temporal windows.
"""

import json
import os
import random
from typing import Dict, List, Optional, Tuple

RANDOM_SEED = 42
DEFAULT_VALID_SUBJECTS = [
    f"SA{i:02d}" for i in range(6, 38) if i != 34
]  # SA06 to SA37, excluding SA34 (31 subjects)


def split_subjects(
    subjects: Optional[List[str]] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = RANDOM_SEED,
    output_path: Optional[str] = "reports/subject_split.json",
) -> Dict[str, List[str]]:
    """
    Split subject IDs into disjoint Train, Validation, and Test sets.

    Parameters
    ----------
    subjects : list of str, optional
        List of subject IDs. Defaults to official KFall subjects SA06-SA37 (excluding SA34).
    train_ratio : float
        Proportion of subjects for training (default 0.70).
    val_ratio : float
        Proportion of subjects for validation (default 0.15).
    test_ratio : float
        Proportion of subjects for test (default 0.15).
    random_seed : int
        Seed for reproducibility (default 42).
    output_path : str, optional
        Path to save the JSON split configuration.

    Returns
    -------
    dict with keys 'train', 'validation', 'test'
    """
    if subjects is None:
        subject_pool = sorted(list(DEFAULT_VALID_SUBJECTS))
    else:
        subject_pool = sorted(list(set(subjects)))

    n_total = len(subject_pool)
    if n_total == 0:
        raise ValueError("Subject pool is empty. Cannot perform split.")

    # Deterministic shuffle
    rng = random.Random(random_seed)
    shuffled = list(subject_pool)
    rng.shuffle(shuffled)

    if n_total < 3:
        # Fallback for very small test sets
        train_subs = shuffled[:1]
        val_subs = shuffled[1:2] if n_total > 1 else []
        test_subs = []
    else:
        n_val = max(1, round(n_total * val_ratio))
        n_test = max(1, round(n_total * test_ratio))
        n_train = n_total - n_val - n_test

        # Ensure train gets at least 1 if n_total >= 3
        if n_train < 1:
            n_train = 1
            if n_val > n_test:
                n_val -= 1
            else:
                n_test -= 1

        train_subs = sorted(shuffled[:n_train])
        val_subs = sorted(shuffled[n_train : n_train + n_val])
        test_subs = sorted(shuffled[n_train + n_val :])

    split_result = {
        "random_seed": random_seed,
        "total_subjects": n_total,
        "train_subjects": train_subs,
        "validation_subjects": val_subs,
        "test_subjects": test_subs,
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(split_result, f, indent=2)

    return split_result


if __name__ == "__main__":
    split_info = split_subjects()
    print(f"Generated Subject Split (Seed {RANDOM_SEED}):")
    print(f"  Train ({len(split_info['train_subjects'])}): {split_info['train_subjects']}")
    print(f"  Validation ({len(split_info['validation_subjects'])}): {split_info['validation_subjects']}")
    print(f"  Test ({len(split_info['test_subjects'])}): {split_info['test_subjects']}")
