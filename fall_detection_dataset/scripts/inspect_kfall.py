"""
Inspect KFall raw sensor files and label excel files.
"""
import glob
import os
import pandas as pd

kfall_root = os.path.join("fall_detection_dataset", "raw", "kfall")
sensor_files = sorted(glob.glob(os.path.join(kfall_root, "sensor_data", "*", "*.csv")))
print(f"Total KFall sensor files found: {len(sensor_files)}")

for sf in sensor_files:
    df = pd.read_csv(sf)
    sub = os.path.basename(os.path.dirname(sf))
    fname = os.path.basename(sf)
    dt = df["TimeStamp(s)"].diff().dropna().mean()
    hz = round(1.0 / dt, 2) if dt > 0 else 0
    print(f"Subject: {sub}, File: {fname}, Rows: {len(df)}, Rate: ~{hz} Hz, Columns: {list(df.columns)}")

label_files = sorted(glob.glob(os.path.join(kfall_root, "label_data", "*.xlsx")))
print(f"\nTotal KFall label files found: {len(label_files)}")
for lf in label_files:
    xl = pd.ExcelFile(lf)
    print(f"\nLabel file: {os.path.basename(lf)}, Sheets: {xl.sheet_names}")
    for sheet in xl.sheet_names:
        df_l = xl.parse(sheet)
        print(f"  Sheet: {sheet}, Shape: {df_l.shape}")
        print(df_l.head(5))
