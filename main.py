"""
main.py
Main execution script for CMPSC60 Project 1:
Time-Series Analysis of Water Pump Sensor Data

Runs all three tasks in order:
  Task 1 - Divide-and-Conquer Segmentation
  Task 2 - Divide-and-Conquer Clustering
  Task 3 - Maximum Subarray (Kadane's Algorithm)

Usage:
    python main.py
"""

import os
import numpy as np

from data_loading import load_data, get_sensor_columns, add_rul_category, summarize_categories
from task1_segmentation import run_task1, toy_example as toy1
from task2_clustering import run_task2, toy_example as toy2
from task3_kadane import run_task3, toy_example as toy3


# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────

DATA_FILE  = 'rul_hrs.csv'
NROWS      = 10_000
N_SENSORS  = 10        # sensors to use for Task 1
RANDOM_SEED = 42
OUT_DIR    = 'plots'

os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  CMPSC60 Project 1: Water Pump Sensor Analysis")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────
    print(f"\nLoading '{DATA_FILE}' (first {NROWS:,} rows)...")
    df = load_data(DATA_FILE, nrows=NROWS)
    print(f"  Shape: {df.shape}")

    df, quantiles = add_rul_category(df)
    summarize_categories(df)

    sensors = get_sensor_columns(df)
    print(f"\nTotal sensor columns: {len(sensors)}")

    # ── Toy examples (for demonstration / report) ──────────────
    print("\n" + "=" * 60)
    print("  TOY EXAMPLES")
    print("=" * 60)
    toy1()
    toy2()
    toy3()

    # ── Task 1 ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    task1_results, selected_sensors = run_task1(
        df, sensors,
        n_sensors=N_SENSORS,
        seed=RANDOM_SEED,
        out_dir=OUT_DIR,
    )

    # ── Task 2 ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    task2_labels = run_task2(df, sensors)

    # ── Task 3 ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    task3_results = run_task3(df, sensors, top_n=10, out_dir=OUT_DIR)

    # ── Final summary ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"\nTask 1: Segmented {N_SENSORS} sensors.")
    scores = {s: task1_results[s]['score'] for s in task1_results}
    most_complex = max(scores, key=scores.get)
    least_complex = min(scores, key=scores.get)
    print(f"  Most complex  : {most_complex} ({scores[most_complex]} segments)")
    print(f"  Least complex : {least_complex} ({scores[least_complex]} segments)")

    unique_clusters = len(np.unique(task2_labels))
    print(f"\nTask 2: Produced {unique_clusters} clusters from {len(df):,} rows.")

    cat_names = {0: 'Extremely Low', 1: 'Moderately Low',
                 2: 'Moderately High', 3: 'Extremely High'}
    low_rul_sensors = [s for s, r in task3_results.items() if r['dominant_rul'] in (0, 1)]
    print(f"\nTask 3: {len(low_rul_sensors)} sensors have max-activity windows in Low RUL regions:")
    for s in low_rul_sensors[:5]:
        print(f"  {s}: {cat_names[task3_results[s]['dominant_rul']]}")

    print(f"\nAll plots saved to '{OUT_DIR}/'")
    print("\nDone.")


if __name__ == '__main__':
    main()
