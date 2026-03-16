# main.py
# runs all three tasks for CMPSC60 Project 1
# Usage: python main.py

import os
import numpy as np

from data_loading import load_data, get_sensor_columns, add_rul_category, summarize_categories
from task1_segmentation import run_task1, toy_example as toy1
from task2_clustering import run_task2, toy_example as toy2
from task3_kadane import run_task3, toy_example as toy3

DATA_FILE = 'rul_hrs.csv'
NROWS = 10000
N_SENSORS = 10
RANDOM_SEED = 42
OUT_DIR = 'plots'

os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("  CMPSC60 Project 1: Water Pump Sensor Analysis")
    print("=" * 60)

    # load data
    print(f"\nLoading '{DATA_FILE}' (first {NROWS} rows)...")
    df = load_data(DATA_FILE, nrows=NROWS)
    print(f"Shape: {df.shape}")

    df, quantiles = add_rul_category(df)
    summarize_categories(df)

    sensors = get_sensor_columns(df)
    print(f"\nTotal sensor columns: {len(sensors)}")

    # show toy examples first so you can verify the algorithms work
    print("\n" + "=" * 60)
    print("  TOY EXAMPLES")
    print("=" * 60)
    toy1()
    toy2()
    toy3()

    # Task 1
    print("\n" + "=" * 60)
    task1_results, selected_sensors = run_task1(
        df, sensors,
        n_sensors=N_SENSORS,
        seed=RANDOM_SEED,
        out_dir=OUT_DIR,
    )

    # Task 2
    print("\n" + "=" * 60)
    task2_labels = run_task2(df, sensors)

    # Task 3
    print("\n" + "=" * 60)
    task3_results = run_task3(df, sensors, top_n=10, out_dir=OUT_DIR)

    # print a quick summary at the end
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    scores = {s: task1_results[s]['score'] for s in task1_results}
    most_complex = max(scores, key=scores.get)
    least_complex = min(scores, key=scores.get)
    print(f"\nTask 1: Segmented {N_SENSORS} sensors.")
    print(f"  Most complex:  {most_complex} ({scores[most_complex]} segments)")
    print(f"  Least complex: {least_complex} ({scores[least_complex]} segments)")

    print(f"\nTask 2: Produced {len(np.unique(task2_labels))} clusters from {len(df)} rows.")

    cat_names = {0: 'Extremely Low', 1: 'Moderately Low',
                 2: 'Moderately High', 3: 'Extremely High'}
    low_rul_sensors = [s for s, r in task3_results.items() if r['dominant_rul'] in (0, 1)]
    print(f"\nTask 3: {len(low_rul_sensors)} sensors have peak deviation in Low RUL regions:")
    for s in low_rul_sensors[:5]:
        print(f"  {s}: {cat_names[task3_results[s]['dominant_rul']]}")

    print(f"\nPlots saved to '{OUT_DIR}/'")
    print("\nDone.")


if __name__ == '__main__':
    main()
