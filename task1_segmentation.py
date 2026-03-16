# task1_segmentation.py
# Divide and conquer segmentation using variance threshold
#
# The idea: if a segment has high variance, split it in half and check each half.
# Keep splitting until variance is low enough (stable segment) or the segment
# is too small to split further.

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import random
import os


def segment_recursive(values, start, end, threshold, min_size, segments):
    # base case: segment too small to split
    if end - start < min_size:
        segments.append((start, end))
        return

    var = np.var(values[start:end])

    if var <= threshold:
        # variance is low enough, mark as stable
        segments.append((start, end))
    else:
        # split in half and recurse on each half
        mid = (start + end) // 2
        segment_recursive(values, start, mid, threshold, min_size, segments)
        segment_recursive(values, mid, end, threshold, min_size, segments)


def segment_sensor(values, threshold=None, min_size=50):
    # use 10% of the overall variance as threshold - seemed to work reasonably well
    if threshold is None:
        threshold = 0.1 * np.var(values)

    segments = []
    segment_recursive(values, 0, len(values), threshold, min_size, segments)
    segments.sort(key=lambda x: x[0])
    return segments


def complexity_score(segments):
    # just the number of segments
    return len(segments)


def plot_segmentation(values, segments, sensor_name, rul_categories, out_dir='plots'):
    os.makedirs(out_dir, exist_ok=True)

    # colors for each RUL category
    cat_colors = {
        0: '#ff9999',  # Extremely Low
        1: '#ffcc99',  # Moderately Low
        2: '#99ccff',  # Moderately High
        3: '#99ff99',  # Extremely High
    }
    cat_labels = {0: 'Ext. Low', 1: 'Mod. Low', 2: 'Mod. High', 3: 'Ext. High'}

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(values, color='steelblue', linewidth=0.6, alpha=0.8)

    for (s, e) in segments:
        # find which RUL category shows up most in this segment
        seg_cats = rul_categories[s:e]
        dominant = int(np.bincount(seg_cats).argmax())
        ax.axvspan(s, e, alpha=0.25, color=cat_colors[dominant])
        ax.axvline(x=s, color='gray', linewidth=0.5, linestyle='--', alpha=0.6)

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


def run_task1(df, sensor_columns, n_sensors=10, seed=42, out_dir='plots'):
    random.seed(seed)
    np.random.seed(seed)
    selected = random.sample(sensor_columns, min(n_sensors, len(sensor_columns)))

    rul_cats = df['rul_category'].values
    results = {}

    print("\n=== TASK 1: Divide-and-Conquer Segmentation ===")
    print(f"Selected sensors: {selected}\n")

    for sensor in selected:
        values = df[sensor].values.astype(float)

        # fill NaN values by carrying the previous value forward
        nan_mask = np.isnan(values)
        if nan_mask.any():
            for i in range(len(values)):
                if nan_mask[i] and i > 0:
                    values[i] = values[i - 1]

        segs = segment_sensor(values)
        score = complexity_score(segs)
        results[sensor] = {'segments': segs, 'score': score}

        path = plot_segmentation(values, segs, sensor, rul_cats, out_dir=out_dir)
        print(f"  {sensor}: {score} segments -> saved {path}")

    print("\nSegmentation Complexity Scores:")
    print(f"  {'Sensor':<15} {'Score':>6}")
    print(f"  {'-'*22}")
    for s, r in sorted(results.items(), key=lambda x: -x[1]['score']):
        print(f"  {s:<15} {r['score']:>6}")

    # check which RUL category shows up most often across segments for each sensor
    print("\nDominant RUL category per sensor:")
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


def toy_example():
    print("\n--- Toy Example (segmentation) ---")
    arr = np.array([1.0, 1.1, 1.0, 1.2, 5.0, 9.0, 8.5, 9.2, 1.1, 1.0])
    print(f"Input: {arr}")
    segs = segment_sensor(arr, threshold=1.0, min_size=2)
    print(f"Segments (threshold=1.0, min_size=2): {segs}")
    for s, e in segs:
        print(f"  [{s}:{e}] = {arr[s:e]}  var={np.var(arr[s:e]):.4f}")


if __name__ == '__main__':
    toy_example()
