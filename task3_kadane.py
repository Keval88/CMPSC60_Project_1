"""
task3_kadane.py
Maximum Subarray via Kadane's Algorithm applied to sensor data.

For each sensor:
  1. Compute absolute first difference: d[i] = |sensor[i] - sensor[i-1]|
     (length = N-1)
  2. Center it: x[i] = d[i] - mean(d)
  3. Apply Kadane's algorithm to find the max-sum subarray
  4. Look at which RUL category dominates the winning interval
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


# ─────────────────────────────────────────────
#  Kadane's Algorithm
# ─────────────────────────────────────────────

def kadane(arr):
    """
    Find the contiguous subarray with the maximum sum.

    Returns
    -------
    max_sum   : float  - value of the maximum subarray sum
    start_idx : int    - start index (inclusive)
    end_idx   : int    - end index (inclusive)
    """
    if len(arr) == 0:
        return 0.0, 0, 0

    max_sum = arr[0]
    current_sum = arr[0]
    best_start = 0
    best_end = 0
    current_start = 0

    for i in range(1, len(arr)):
        if current_sum + arr[i] < arr[i]:
            current_sum = arr[i]
            current_start = i
        else:
            current_sum += arr[i]

        if current_sum > max_sum:
            max_sum = current_sum
            best_start = current_start
            best_end = i

    return float(max_sum), int(best_start), int(best_end)


# ─────────────────────────────────────────────
#  Per-sensor analysis
# ─────────────────────────────────────────────

def analyze_sensor(values, rul_categories):
    """
    Run the full pipeline for one sensor.

    Parameters
    ----------
    values         : 1-D numpy array of raw sensor readings (length N)
    rul_categories : 1-D int array of RUL categories (length N)

    Returns
    -------
    dict with keys: max_sum, start, end, dominant_rul, diff, centered
    """
    # step 1: absolute first difference
    diff = np.abs(np.diff(values.astype(float)))   # length N-1

    # step 2: center
    centered = diff - np.mean(diff)

    # step 3: Kadane
    max_sum, start, end = kadane(centered)

    # step 4: dominant RUL in the interval
    # diff[i] corresponds to the transition from row i to i+1,
    # so map back to original rows [start, end+1]
    seg_cats = rul_categories[start: end + 2]   # +2 because diff is shifted by 1
    if len(seg_cats) == 0:
        dominant = -1
    else:
        dominant = int(np.bincount(seg_cats, minlength=4).argmax())

    return {
        'max_sum': max_sum,
        'start': start,
        'end': end,
        'dominant_rul': dominant,
        'diff': diff,
        'centered': centered,
    }


# ─────────────────────────────────────────────
#  Run Task 3
# ─────────────────────────────────────────────

def run_task3(df, sensor_columns, top_n=10, out_dir='plots'):
    """
    Apply Kadane's analysis to all sensors. Print results and identify
    sensors that are good early indicators of low RUL.

    Returns
    -------
    results : dict  sensor_name -> analysis dict
    """
    os.makedirs(out_dir, exist_ok=True)
    rul_cats = df['rul_category'].values

    print("\n=== TASK 3: Kadane's Algorithm on Sensor Differences ===")

    results = {}
    for sensor in sensor_columns:
        vals = df[sensor].values.astype(float)
        # handle NaN
        mask = np.isnan(vals)
        for i in range(len(vals)):
            if mask[i] and i > 0:
                vals[i] = vals[i - 1]
        results[sensor] = analyze_sensor(vals, rul_cats)

    cat_names = {0: 'Extremely Low', 1: 'Moderately Low',
                 2: 'Moderately High', 3: 'Extremely High'}

    # print summary table
    # note: max_sum = total deviation of the max-sum subarray (assignment terminology)
    print(f"\n  {'Sensor':<15} {'Total Deviation':>16}  {'Start':>7}  {'End':>7}  {'Dominant RUL':<20}")
    print(f"  {'-'*74}")
    for sensor, r in results.items():
        cat = cat_names.get(r['dominant_rul'], 'Unknown')
        print(f"  {sensor:<15} {r['max_sum']:>16.4f}  {r['start']:>7}  {r['end']:>7}  {cat:<20}")

    # sensors where the dominant RUL is Extremely Low or Moderately Low
    low_rul_sensors = [s for s, r in results.items() if r['dominant_rul'] in (0, 1)]
    print(f"\nSensors with max-deviation interval dominated by LOW RUL ({len(low_rul_sensors)}):")
    for s in low_rul_sensors:
        r = results[s]
        print(f"  {s}: dominant={cat_names[r['dominant_rul']]}  interval=[{r['start']}, {r['end']}]  total_deviation={r['max_sum']:.4f}")

    if low_rul_sensors:
        print("\n  -> These sensors show high variability precisely when RUL is low,")
        print("     making them good early-warning indicators of pump failure.")
    else:
        print("\n  -> No sensor's peak activity aligns with low RUL in this dataset.")

    # plot top_n sensors by max_sum
    sorted_sensors = sorted(results.items(), key=lambda x: -x[1]['max_sum'])[:top_n]
    _plot_top_sensors(sorted_sensors, cat_names, out_dir)

    return results


def _plot_top_sensors(sorted_sensors, cat_names, out_dir):
    """Save a summary plot of the top sensors by Kadane max sum."""
    fig, axes = plt.subplots(len(sorted_sensors), 1,
                             figsize=(14, 3 * len(sorted_sensors)))
    if len(sorted_sensors) == 1:
        axes = [axes]

    cat_colors = {0: 'red', 1: 'orange', 2: 'steelblue', 3: 'green', -1: 'gray'}

    for ax, (sensor, r) in zip(axes, sorted_sensors):
        centered = r['centered']
        s, e = r['start'], r['end']
        dom = r['dominant_rul']

        ax.plot(centered, color='steelblue', linewidth=0.7, alpha=0.7)
        ax.axvspan(s, e, alpha=0.35, color=cat_colors.get(dom, 'gray'),
                   label=f"Max subarray: {cat_names.get(dom, '?')}")
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.set_title(f"{sensor}  |  max_sum={r['max_sum']:.3f}  interval=[{s},{e}]", fontsize=9)
        ax.set_ylabel("Centered |diff|", fontsize=8)
        ax.legend(fontsize=8, loc='upper right')

    axes[-1].set_xlabel("Time index (shifted by 1)")
    plt.tight_layout()
    path = os.path.join(out_dir, "kadane_top_sensors.png")
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"\n  Saved Kadane plot -> {path}")


# ─────────────────────────────────────────────
#  Toy example (for report / demo)
# ─────────────────────────────────────────────

def toy_example():
    """Demonstrate Kadane's on a small array."""
    print("\n--- Toy Example (Kadane's Algorithm) ---")
    arr = np.array([-2.0, 1.0, -3.0, 4.0, -1.0, 2.0, 1.0, -5.0, 4.0])
    print(f"Input: {arr}")
    max_sum, start, end = kadane(arr)
    print(f"Max subarray sum: {max_sum}")
    print(f"Subarray indices: [{start}, {end}]  -> {arr[start:end+1]}")

    # also show the pipeline
    print("\n--- Toy Pipeline ---")
    sensor = np.array([10.0, 10.1, 10.2, 15.0, 20.0, 19.8, 10.3, 10.1, 10.2])
    rul_cats = np.array([3, 3, 3, 2, 1, 0, 0, 0, 0])
    print(f"Sensor values: {sensor}")
    result = analyze_sensor(sensor, rul_cats)
    print(f"Abs diff:      {result['diff']}")
    print(f"Centered:      {result['centered'].round(4)}")
    print(f"Max sum: {result['max_sum']:.4f}  interval=[{result['start']},{result['end']}]")
    cat_names = {0: 'Extremely Low', 1: 'Moderately Low',
                 2: 'Moderately High', 3: 'Extremely High'}
    print(f"Dominant RUL category in interval: {cat_names[result['dominant_rul']]}")


if __name__ == '__main__':
    toy_example()
