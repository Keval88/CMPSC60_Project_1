# task3_kadane.py
# Kadane's algorithm applied to sensor time series
#
# For each sensor we:
#   1. compute absolute first difference: d[i] = |sensor[i] - sensor[i-1]|
#   2. subtract the mean so values above average are positive, below are negative
#   3. run Kadane's to find the window with the most "excess" variability
#   4. check which RUL category dominates that window

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def kadane(arr):
    # standard Kadane's algorithm - O(n)
    # returns (max_sum, start_index, end_index)
    if len(arr) == 0:
        return 0.0, 0, 0

    max_sum = arr[0]
    cur_sum = arr[0]
    best_start = 0
    best_end = 0
    cur_start = 0

    for i in range(1, len(arr)):
        if cur_sum + arr[i] < arr[i]:
            # starting fresh is better
            cur_sum = arr[i]
            cur_start = i
        else:
            cur_sum += arr[i]

        if cur_sum > max_sum:
            max_sum = cur_sum
            best_start = cur_start
            best_end = i

    return float(max_sum), int(best_start), int(best_end)


def analyze_sensor(values, rul_categories):
    # step 1: absolute first difference
    diff = np.abs(np.diff(values.astype(float)))

    # step 2: subtract mean to center it
    centered = diff - np.mean(diff)

    # step 3: run Kadane's to find the highest-deviation window
    total_deviation, start, end = kadane(centered)

    # step 4: figure out which RUL category dominates that window
    # diff[i] is the change between row i and row i+1, so add 1 to map back
    seg_cats = rul_categories[start: end + 2]
    if len(seg_cats) == 0:
        dominant = -1
    else:
        dominant = int(np.bincount(seg_cats, minlength=4).argmax())

    return {
        'total_deviation': total_deviation,
        'max_sum': total_deviation,  # keeping max_sum as alias so nothing breaks
        'start': start,
        'end': end,
        'dominant_rul': dominant,
        'diff': diff,
        'centered': centered,
    }


def run_task3(df, sensor_columns, top_n=10, out_dir='plots'):
    os.makedirs(out_dir, exist_ok=True)
    rul_cats = df['rul_category'].values

    print("\n=== TASK 3: Kadane's Algorithm on Sensor Differences ===")

    results = {}
    for sensor in sensor_columns:
        vals = df[sensor].values.astype(float)
        # fill NaN by carrying previous value forward
        nan_mask = np.isnan(vals)
        for i in range(len(vals)):
            if nan_mask[i] and i > 0:
                vals[i] = vals[i - 1]
        results[sensor] = analyze_sensor(vals, rul_cats)

    cat_names = {0: 'Extremely Low', 1: 'Moderately Low',
                 2: 'Moderately High', 3: 'Extremely High'}

    print(f"\n  {'Sensor':<15} {'Total Deviation':>16}  {'Start':>7}  {'End':>7}  {'Dominant RUL':<20}")
    print(f"  {'-'*74}")
    for sensor, r in results.items():
        cat = cat_names.get(r['dominant_rul'], 'Unknown')
        print(f"  {sensor:<15} {r['total_deviation']:>16.4f}  {r['start']:>7}  {r['end']:>7}  {cat:<20}")

    low_rul_sensors = [s for s, r in results.items() if r['dominant_rul'] in (0, 1)]
    print(f"\nSensors with max-deviation interval in LOW RUL ({len(low_rul_sensors)}):")
    for s in low_rul_sensors:
        r = results[s]
        print(f"  {s}: dominant={cat_names[r['dominant_rul']]}  interval=[{r['start']}, {r['end']}]  total_deviation={r['total_deviation']:.4f}")

    if low_rul_sensors:
        print("\n  -> These sensors are most variable when RUL is low,")
        print("     so they could work as early warning indicators of pump failure.")
    else:
        print("\n  -> No sensor's peak activity lines up with low RUL.")

    # save a plot for the top N sensors
    sorted_sensors = sorted(results.items(), key=lambda x: -x[1]['total_deviation'])[:top_n]
    plot_top_sensors(sorted_sensors, cat_names, out_dir)

    return results


def plot_top_sensors(sorted_sensors, cat_names, out_dir):
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
                   label=f"Max window: {cat_names.get(dom, '?')}")
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.set_title(f"{sensor}  |  total_deviation={r['total_deviation']:.3f}  interval=[{s},{e}]", fontsize=9)
        ax.set_ylabel("Centered |diff|", fontsize=8)
        ax.legend(fontsize=8, loc='upper right')

    axes[-1].set_xlabel("Time index")
    plt.tight_layout()
    path = os.path.join(out_dir, "kadane_top_sensors.png")
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"\n  Saved Kadane plot -> {path}")


def toy_example():
    print("\n--- Toy Example (Kadane's Algorithm) ---")
    arr = np.array([-2.0, 1.0, -3.0, 4.0, -1.0, 2.0, 1.0, -5.0, 4.0])
    print(f"Input: {arr}")
    total_dev, start, end = kadane(arr)
    print(f"Total deviation: {total_dev}")
    print(f"Subarray indices: [{start}, {end}] -> {arr[start:end+1]}")

    print("\n--- Toy Pipeline ---")
    sensor = np.array([10.0, 10.1, 10.2, 15.0, 20.0, 19.8, 10.3, 10.1, 10.2])
    rul_cats = np.array([3, 3, 3, 2, 1, 0, 0, 0, 0])
    print(f"Sensor values: {sensor}")
    result = analyze_sensor(sensor, rul_cats)
    print(f"Abs diff:       {result['diff']}")
    print(f"Centered:       {result['centered'].round(4)}")
    print(f"Total deviation: {result['total_deviation']:.4f}  interval=[{result['start']},{result['end']}]")
    cat_names = {0: 'Extremely Low', 1: 'Moderately Low',
                 2: 'Moderately High', 3: 'Extremely High'}
    print(f"Dominant RUL in interval: {cat_names[result['dominant_rul']]}")


if __name__ == '__main__':
    toy_example()
