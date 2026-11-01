pass
import random

import numpy as np
from sklearn.model_selection import train_test_split


BOOTSTRAP_SEEDS = tuple(range(10))
TEST_FRACTION = 0.20
TEST_GROUP_SIZE = 10
VAL_FRACTION = 0.15


def split_domain_indices(domain_indices, split_seed=42, bool_bootstrap=True):
    pass
    rng = random.Random(split_seed)
    splits = {}
    for domain, source_ids in domain_indices.items():
        ids = list(source_ids)
        n = len(ids)
        if not bool_bootstrap:
            if n < 3:
                raise ValueError(
                    f"Domain {domain} has {n} molecules; at least 3 are needed "
                    "for disjoint train, validation, and test sets."
                )
            if n >= 6:
                train_ids, test_val_ids = train_test_split(
                    ids, test_size=0.2, random_state=split_seed
                )
                val_ids, test_ids = train_test_split(
                    test_val_ids, test_size=0.5, random_state=split_seed
                )
            else:
                small_split_ids = list(ids)
                random.Random(split_seed).shuffle(small_split_ids)
                train_ids = small_split_ids[:-2]
                val_ids = small_split_ids[-2:-1]
                test_ids = small_split_ids[-1:]
            test_pool = list(test_ids)
            groups = {}
        else:
            rng.shuffle(ids)
            if n < 3:


                train_ids = list(ids)
                val_ids = list(ids)
                test_pool = list(ids)
            else:
                n_test = max(1, int(round(TEST_FRACTION * n)))
                n_val = max(1, int(round(VAL_FRACTION * n)))
                while n_test + n_val >= n:
                    if n_val >= n_test and n_val > 1:
                        n_val -= 1
                    elif n_test > 1:
                        n_test -= 1
                    else:
                        break
                train_end = n - n_val - n_test
                train_ids = ids[:train_end]
                val_ids = ids[train_end:train_end + n_val]
                test_pool = ids[train_end + n_val:]
            group_size = TEST_GROUP_SIZE
            groups = {
                seed: np.random.RandomState(seed).choice(
                    test_pool, size=group_size, replace=True
                ).astype(int).tolist()
                for seed in BOOTSTRAP_SEEDS
            } if test_pool else {seed: [] for seed in BOOTSTRAP_SEEDS}
            test_ids = test_pool
        splits[domain] = {
            "train": train_ids,
            "val": val_ids,
            "test_pool": test_pool,
            "test": test_ids,
            "test_groups": groups,
            "group_size": TEST_GROUP_SIZE if bool_bootstrap else len(test_ids),
        }
    return splits


def aggregate_group_metrics(group_metrics):
    pass
    summary = {}
    for key in ("loss", "accuracy", "auc", "precision", "recall", "f1"):
        values = [m[key] for m in group_metrics
                  if m.get(key) is not None]
        if values:
            summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=0)),
            }
    return summary

