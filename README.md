# CMPSC60 Project 1 — Water Pump Sensor Time-Series Analysis

Algorithms course project implementing divide-and-conquer segmentation, divisive clustering,
and Kadane's maximum-subarray algorithm on water pump sensor data.

## Dataset

Water Pump RUL dataset (`rul_hrs.csv`) — first 10,000 rows, 50 sensor columns + RUL.

Download from: https://www.kaggle.com/datasets/anseldsouza/water-pump-rul-predictive-maintenance

Place `rul_hrs.csv` in the project root before running.

## Requirements

```
pandas
numpy
matplotlib
```

Install with:

```bash
pip install pandas numpy matplotlib
```

## Project Structure

```
CMPSC60_Project_1/
├── rul_hrs.csv           # dataset (download separately)
├── data_loading.py       # load CSV, add RUL category column
├── task1_segmentation.py # Task 1: divide-and-conquer variance segmentation
├── task2_clustering.py   # Task 2: divisive clustering into 4 clusters
├── task3_kadane.py       # Task 3: Kadane's max-subarray on sensor diffs
├── main.py               # runs all tasks end-to-end
├── plots/                # output plots (created automatically)
└── README.md
```

## How to Run

```bash
python main.py
```

Output goes to the terminal and plots are saved to `plots/`.

## Tasks

### Task 1 — Divide-and-Conquer Segmentation
- Randomly selects 10 sensors
- Recursively splits each sensor's time series when variance exceeds a threshold
- Plots each sensor with color-coded RUL segments
- Reports a "complexity score" (number of segments)

### Task 2 — Divisive Clustering
- Clusters all 10,000 rows (50 sensor features) into 4 clusters
- Uses recursive bisection on the highest-variance feature
- Reports cluster size, majority RUL category, and purity

### Task 3 — Kadane's Algorithm
- For each sensor: computes |first difference|, centers it, then runs Kadane's
- Identifies sensors whose max-activity window falls in low RUL regions
- These sensors are candidate early-warning indicators of pump failure

## Algorithms Used
- **Divide and Conquer** — recursive variance-based segmentation (Task 1)
- **Divisive Clustering** — top-down cluster bisection (Task 2)
- **Kadane's Algorithm** — O(n) maximum subarray (Task 3)

No machine learning libraries (scikit-learn, tensorflow, etc.) are used.
