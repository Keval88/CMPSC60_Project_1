# task2_clustering.py
# Divide and conquer clustering (divisive / top-down approach)
# No sklearn or any ML library used - built from scratch
#
# How it works:
#   Start with all points in one cluster.
#   Find the cluster with the most spread (highest total variance).
#   Split it by finding the feature with the highest variance, then split at the median.
#   Repeat until we have k=4 clusters.

import numpy as np


def get_spread(X):
    # measure how spread out a cluster is - just sum of variances across all features
    if len(X) <= 1:
        return 0.0
    return float(np.sum(np.var(X, axis=0)))


def bisect_cluster(X, indices):
    # split this cluster into two groups
    chunk = X[indices]

    # find the feature with the most variance to split on
    variances = np.var(chunk, axis=0)
    split_feature = int(np.argmax(variances))

    median_val = np.median(chunk[:, split_feature])

    left_mask = chunk[:, split_feature] <= median_val

    # edge case: if all values are the same, just split in half
    if left_mask.all() or (~left_mask).all():
        mid = len(indices) // 2
        return indices[:mid], indices[mid:]

    return indices[left_mask], indices[~left_mask]


def divisive_cluster(X, k=4):
    # start with everything in one cluster
    clusters = [np.arange(len(X))]

    while len(clusters) < k:
        # pick the most spread out cluster to split next
        spreads = [get_spread(X[c]) for c in clusters]
        worst_idx = int(np.argmax(spreads))

        if spreads[worst_idx] == 0.0:
            # nothing left to split
            break

        left, right = bisect_cluster(X, clusters[worst_idx])
        clusters.pop(worst_idx)
        clusters.append(left)
        clusters.append(right)

    # assign integer labels 0, 1, 2, 3
    labels = np.empty(len(X), dtype=int)
    for label, idx_arr in enumerate(clusters):
        labels[idx_arr] = label

    return labels


def run_task2(df, sensor_columns):
    print("\n=== TASK 2: Divide-and-Conquer Clustering ===")

    X = df[sensor_columns].values.astype(float)

    # replace NaN with the column mean
    col_means = np.nanmean(X, axis=0)
    for j in range(X.shape[1]):
        nan_mask = np.isnan(X[:, j])
        X[nan_mask, j] = col_means[j]

    print(f"Running divisive clustering on {X.shape[0]} rows x {X.shape[1]} features...")
    labels = divisive_cluster(X, k=4)

    rul_cats = df['rul_category'].values
    cat_names = {0: 'Extremely Low', 1: 'Moderately Low',
                 2: 'Moderately High', 3: 'Extremely High'}

    print("\nCluster Summary:")
    print(f"  {'Cluster':<10} {'Size':>7}  {'Majority RUL Category':<22}  {'Majority Count':>14}  {'Purity':>7}")
    print(f"  {'-'*65}")

    for cid in range(4):
        mask = labels == cid
        size = int(mask.sum())
        if size == 0:
            print(f"  {cid:<10} {0:>7}  {'(empty)':<22}")
            continue
        cats_in_cluster = rul_cats[mask]
        counts = np.bincount(cats_in_cluster, minlength=4)
        majority_cat = int(np.argmax(counts))
        majority_count = int(counts[majority_cat])
        purity = majority_count / size
        print(f"  {cid:<10} {size:>7}  {cat_names[majority_cat]:<22}  {majority_count:>14}  {purity:>6.1%}")

    # overall purity
    total_majority = sum(
        int(np.bincount(rul_cats[labels == cid], minlength=4).max())
        for cid in range(4)
        if (labels == cid).sum() > 0
    )
    overall_purity = total_majority / len(labels)
    print(f"\n  Overall clustering purity: {overall_purity:.1%}")

    print("\nDiscussion:")
    print("  The divisive clustering groups rows by sensor behavior patterns.")
    print("  Clusters with high spread tend to capture degraded pump states")
    print("  (lower RUL), while uniform clusters correspond to stable operation.")

    return labels


def toy_example():
    print("\n--- Toy Example (divisive clustering) ---")
    X = np.array([
        [1.0, 2.0],
        [1.5, 1.8],
        [1.2, 2.2],
        [8.0, 9.0],
        [8.5, 8.8],
        [9.0, 9.5],
        [1.1, 8.0],
        [1.3, 7.8],
    ])
    print(f"Input (8 points, 2 features):\n{X}")
    labels = divisive_cluster(X, k=4)
    print(f"Cluster labels: {labels}")
    for cid in range(4):
        pts = X[labels == cid]
        print(f"  Cluster {cid}: {len(pts)} points -> {pts.tolist()}")


if __name__ == '__main__':
    toy_example()
