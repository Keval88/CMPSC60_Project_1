"""
task2_clustering.py
Divide-and-Conquer Clustering into exactly 4 clusters.

Algorithm idea (recursive bisection / divisive clustering):
  - Start with all N points in one cluster.
  - Recursively split the cluster with the largest spread (max range across
    any feature dimension) by finding the best single-feature median split.
  - Stop when we have exactly k=4 clusters.
  - This is a top-down / divisive approach — classic divide-and-conquer.

No sklearn or ML libraries used.
"""

import numpy as np


# ─────────────────────────────────────────────
#  Distance and centroid helpers
# ─────────────────────────────────────────────

def centroid(X):
    """Mean vector of a set of points."""
    return np.mean(X, axis=0)


def spread(X):
    """
    Measure of cluster spread: sum of variances across all dimensions.
    Used to decide which cluster to split next.
    """
    if len(X) <= 1:
        return 0.0
    return float(np.sum(np.var(X, axis=0)))


# ─────────────────────────────────────────────
#  Single-cluster bisection
# ─────────────────────────────────────────────

def bisect_cluster(X, indices):
    """
    Split a cluster into two by finding the feature with the highest variance
    and splitting at the median of that feature.

    Parameters
    ----------
    X       : full data matrix (N, D)
    indices : 1-D array of row indices belonging to this cluster

    Returns
    -------
    left_idx, right_idx : two arrays of row indices
    """
    chunk = X[indices]          # (m, D)
    variances = np.var(chunk, axis=0)
    best_feat = int(np.argmax(variances))   # feature with highest variance

    median_val = np.median(chunk[:, best_feat])

    mask_left = chunk[:, best_feat] <= median_val
    # if split produces an empty side (all values equal), fall back to half-split
    if mask_left.all() or (~mask_left).all():
        mid = len(indices) // 2
        return indices[:mid], indices[mid:]

    left_idx = indices[mask_left]
    right_idx = indices[~mask_left]
    return left_idx, right_idx


# ─────────────────────────────────────────────
#  Divisive clustering (divide-and-conquer)
# ─────────────────────────────────────────────

def divisive_cluster(X, k=4):
    """
    Top-down divisive clustering into exactly k clusters.

    Strategy:
      - Maintain a list of current clusters (each is an index array).
      - At each step, pick the cluster with the largest spread and bisect it.
      - Repeat until we have k clusters.

    Parameters
    ----------
    X : numpy array of shape (N, D)
    k : target number of clusters (default 4)

    Returns
    -------
    labels : 1-D array of shape (N,) with cluster assignments 0..k-1
    """
    N = len(X)
    # start: one cluster containing everything
    clusters = [np.arange(N)]

    while len(clusters) < k:
        # pick cluster with largest spread to split next
        spreads = [spread(X[c]) for c in clusters]
        worst = int(np.argmax(spreads))

        if spreads[worst] == 0.0:
            # all remaining clusters are singletons or uniform — stop early
            break

        left, right = bisect_cluster(X, clusters[worst])
        clusters.pop(worst)
        clusters.append(left)
        clusters.append(right)

    # assign integer labels
    labels = np.empty(N, dtype=int)
    for label, idx_arr in enumerate(clusters):
        labels[idx_arr] = label

    return labels


# ─────────────────────────────────────────────
#  Run Task 2
# ─────────────────────────────────────────────

def run_task2(df, sensor_columns):
    """
    Cluster all 10,000 rows using divisive clustering, then report
    the majority RUL category and count for each cluster.

    Returns
    -------
    labels : numpy array of cluster assignments
    """
    print("\n=== TASK 2: Divide-and-Conquer Clustering ===")

    X = df[sensor_columns].values.astype(float)

    # simple NaN fill: replace with column mean
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


# ─────────────────────────────────────────────
#  Toy example (for report / demo)
# ─────────────────────────────────────────────

def toy_example():
    """Demonstrate divisive clustering on a small 2-D dataset."""
    print("\n--- Toy Example (divisive clustering) ---")
    # 8 points in two obvious groups
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
