# data_loading.py
# loads the dataset and converts RUL into 4 categories

import pandas as pd
import numpy as np


def load_data(filepath, nrows=10000):
    # read only first 10000 rows
    df = pd.read_csv(filepath, nrows=nrows)
    # the csv has an unnamed index column, drop it
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    df = df.reset_index(drop=True)
    return df


def get_sensor_columns(df):
    # return all columns that are sensor readings (not timestamp, rul, or the category columns we add)
    skip = {'timestamp', 'rul', 'rul_category', 'rul_category_name'}
    return [c for c in df.columns if c not in skip]


def add_rul_category(df):
    # compute the three quantile cutoffs
    rul = df['rul'].values
    q10 = np.percentile(rul, 10)
    q50 = np.percentile(rul, 50)
    q90 = np.percentile(rul, 90)

    print(f"RUL quantiles -> Q10: {q10:.2f}, Q50: {q50:.2f}, Q90: {q90:.2f}")

    # assign category 0-3 based on where the rul value falls
    categories = []
    for v in rul:
        if v < q10:
            categories.append(0)   # Extremely Low
        elif v < q50:
            categories.append(1)   # Moderately Low
        elif v < q90:
            categories.append(2)   # Moderately High
        else:
            categories.append(3)   # Extremely High

    df = df.copy()
    df['rul_category'] = categories
    df['rul_category_name'] = df['rul_category'].map({
        0: 'Extremely Low',
        1: 'Moderately Low',
        2: 'Moderately High',
        3: 'Extremely High',
    })
    return df, (q10, q50, q90)


def summarize_categories(df):
    print("\nRUL Category Distribution:")
    counts = df['rul_category_name'].value_counts()
    total = len(df)
    for name, cnt in counts.items():
        print(f"  {name}: {cnt} rows ({100*cnt/total:.1f}%)")


if __name__ == '__main__':
    df = load_data('rul_hrs.csv')
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    df, quantiles = add_rul_category(df)
    summarize_categories(df)
    sensors = get_sensor_columns(df)
    print(f"Sensor columns ({len(sensors)}): {sensors[:5]} ...")
