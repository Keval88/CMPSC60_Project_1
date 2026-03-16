"""
task1_segmentation.py
Divide-and-Conquer Segmentation based on variance threshold.

Algorithm:
    segment(array, start, end):
        if variance(array[start:end]) > threshold:
            mid = (start + end) // 2
            segment(array, start, mid)
            segment(array, mid, end)
        else:
            mark [start, end] as a stable segment

The recursion stops when a segment's variance is below the threshold
or the segment is too small to split further (min_size).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import os


# ─────────────────────────────────────────────
#  Core recursive segmentation
# ─────────────────────────────────────────────

def _segment_recursive(values, start, end, threshold, min_size, segments):
    """
    Recursively split [start, end) if variance > threshold.
    Appends (start, end) tuples to `segments` list for stable regions.
    """
    if end - start < min_size:
        segments.append((start, end))
        return

    chunk = values[start:end]
    var = np.var(chunk)

    if var <= threshold:
        segments.append((start, end))
    else:
        mid = (start + end) // 2
        _segment_recursive(values, start, mid, threshold, min_size, segments)
        _segment_recursive(values, mid, end, threshold, min_size, segments)


def segment_sensor(values, threshold=None, min_size=50):
    """
    Run divide-and-conquer segmentation on a 1-D array.

    Parameters
    ----------
    values    : 1-D numpy array of sensor readings
    threshold : variance threshold; if None, uses 10% of overall variance
    min_size  : minimum segment length before we stop splitting

    Returns
    -------
    segments : list of (start, end) index pairs (end is exclusive)
    """
    if threshold is None:
        threshold = 0.1 * np.var(values)

    segments = []
    _segment_recursive(values, 0, len(values), threshold, min_size, segments)
    # sort by start index (recursion may interleave on edge cases)
    segments.sort(key=lambda x: x[0])
    return segments


def segmentation_complexity_score(segments):
    """Number of segments = complexity score."""
    return len(segments)


# ─────────────────────────────────────────────
#  Visualization
# ─────────────────────────────────────────────

def plot_segmentation(values, segments, sensor_name, rul_categories, out_dir='plots'):
    """
    Line plot of sensor values with vertical lines at segment boundaries.
    Background color encodes the dominant RUL category of each segment.
    """
    os.makedirs(out_dir, exist_ok=True)

    cat_colors = {
        0: '#ff9999',   # Extremely Low  - red-ish
        1: '#ffcc99',   # Moderately Low - orange-ish
        2: '#99ccff',   # Moderately High- blue-ish
        3: '#99ff99',   # Extremely High - green-ish
    }
    cat_labels = {
        0: 'Ext. Low',
        1: 'Mod. Low',
        2: 'Mod. High',
        3: 'Ext. High',
    }

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(values, color='steelblue', linewidth=0.6, alpha=0.8)

    for (s, e) in segments:
        # dominant RUL category in this segment
        seg_cats = rul_categories[s:e]
        dominant = int(np.bincount(seg_cats).argmax())
        ax.axvspan(s, e, alpha=0.25, color=cat_colors[dominant])
        ax.axvline(x=s, color='gray', linewidth=0.5, linestyle='--', alpha=0.6)

    # legend patches
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=cat_colors[k], alpha=0.5, label=cat_labels[k])
                       for k in cat_colors]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    ax.set_title(f"{sensor_name}  |  segments: {len(segments)}")
    ax.set_xlabel("Time index")
    ax.set_ylabel("Sensor value")
    plt.tight_layout()
    path = os.path.join(out_dir, f"seg_{sensor_name}.png")
    plt.savefig(path, dpi=100)
    plt.close()
    return path


# ─────────────────────────────────────────────
#  Run Task 1
# ─────────────────────────────────────────────

def run_task1(df, sensor_columns, n_sensors=10, seed=42, out_dir='plots'):
    """
    Randomly select n_sensors, segment each, plot, and return complexity scores.

    Returns
    -------
    results : dict  sensor_name -> {'segments': [...], 'score': int}
    selected_sensors : list of chosen sensor names
    """
    random.seed(seed)
    np.random.seed(seed)
    selected = random.sample(sensor_columns, min(n_sensors, len(sensor_columns)))

    rul_cats = df['rul_category'].values
    results = {}

    print("\n=== TASK 1: Divide-and-Conquer Segmentation ===")
    print(f"Selected sensors: {selected}\n")

    for sensor in selected:
        values = df[sensor].values.astype(float)

        # handle NaNs by forward-filling
        mask = np.isnan(values)
        if mask.any():
            for i in range(len(values)):
                if mask[i] and i > 0:
                    values[i] = values[i - 1]

        segs = segment_sensor(values)
        score = segmentation_complexity_score(segs)
        results[sensor] = {'segments': segs, 'score': score}

        path = plot_segmentation(values, segs, sensor, rul_cats, out_dir=out_dir)
        print(f"  {sensor}: {score} segments  -> saved {path}")

    # summary table
    print("\nSegmentation Complexity Scores:")
    print(f"  {'Sensor':<15} {'Score':>6}")
    print(f"  {'-'*22}")
    for s, r in sorted(results.items(), key=lambda x: -x[1]['score']):
        print(f"  {s:<15} {r['score']:>6}")

    # relate to RUL categories
    print("\nDominant RUL category per segment (across all selected sensors):")
    cat_names = {0: 'Ext.Low', 1: 'Mod.Low', 2: 'Mod.High', 3: 'Ext.High'}
    for sensor, r in results.items():
        cat_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for (s, e) in r['segments']:
            seg_cats = rul_cats[s:e]
            dom = int(np.bincount(seg_cats).argmax())
            cat_counts[dom] += 1
        most = max(cat_counts, key=cat_counts.get)
        print(f"  {sensor:<15}: mostly {cat_names[most]} segments")

    return results, selected


# ─────────────────────────────────────────────
#  Toy example (for report / demo)
# ─────────────────────────────────────────────

def toy_example():
    """Show segmentation on a small 10-element array."""
    print("\n--- Toy Example (segmentation) ---")
    arr = np.array([1.0, 1.1, 1.0, 1.2, 5.0, 9.0, 8.5, 9.2, 1.1, 1.0])
    print(f"Input: {arr}")
    segs = segment_sensor(arr, threshold=1.0, min_size=2)
    print(f"Segments (threshold=1.0, min_size=2): {segs}")
    for s, e in segs:
        print(f"  [{s}:{e}] = {arr[s:e]}  var={np.var(arr[s:e]):.4f}")


if __name__ == '__main__':
    toy_example()
