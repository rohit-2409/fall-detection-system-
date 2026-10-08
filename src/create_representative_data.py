"""
Helper script to generate representative KFall sensor recordings and label files
to test and verify the Stage 2 data preprocessing pipeline per Rule 9.
"""

import os
import numpy as np
import pandas as pd


def generate_motion_recording(
    n_samples: int = 500,
    is_fall: bool = False,
    fall_onset: int = 150,
    fall_impact: int = 230,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generates a realistic 100 Hz KFall recording with realistic kinematics.
    """
    rng = np.random.RandomState(seed)
    time_stamp = np.arange(n_samples) * 0.010  # 100 Hz, 10ms per sample
    frame_counter = np.arange(1, n_samples + 1)

    # Base resting state: AccZ ~ 1.0g (standing upright), low gyroscope noise
    acc_x = rng.normal(0.02, 0.03, n_samples)
    acc_y = rng.normal(0.01, 0.03, n_samples)
    acc_z = rng.normal(0.98, 0.03, n_samples)

    gyr_x = rng.normal(0.0, 4.0, n_samples)
    gyr_y = rng.normal(0.0, 4.0, n_samples)
    gyr_z = rng.normal(0.0, 4.0, n_samples)

    euler_x = rng.normal(0.0, 1.0, n_samples)
    euler_y = rng.normal(0.0, 1.0, n_samples)
    euler_z = rng.normal(0.0, 1.0, n_samples)

    if not is_fall:
        # Normal ADL: gentle dynamic motions (e.g. walking, bending)
        t = np.linspace(0, 4 * np.pi, n_samples)
        acc_z += 0.15 * np.sin(t)
        acc_x += 0.08 * np.cos(t)
        gyr_x += 25.0 * np.sin(t)
        gyr_y += 20.0 * np.cos(t)
    else:
        # Fall dynamics:
        # 1. Before onset: normal walking/standing
        t_pre = np.linspace(0, 2 * np.pi, fall_onset)
        acc_z[:fall_onset] += 0.1 * np.sin(t_pre)

        # 2. Onset to Impact (Loss of balance & Free fall)
        fall_phase_len = fall_impact - fall_onset
        if fall_phase_len > 0:
            phase_ratio = np.linspace(0, np.pi, fall_phase_len)
            # Free fall weightlessness: acceleration dips sharply
            acc_z[fall_onset:fall_impact] = 0.25 + 0.15 * np.cos(phase_ratio)
            acc_x[fall_onset:fall_impact] = 0.60 * np.sin(phase_ratio)
            # Rapid rotational turnover
            gyr_y[fall_onset:fall_impact] = 280.0 * np.sin(phase_ratio)
            gyr_x[fall_onset:fall_impact] = 150.0 * np.sin(phase_ratio)

        # 3. Impact peak: severe acceleration spike
        impact_window = 10
        imp_end = min(n_samples, fall_impact + impact_window)
        imp_len = imp_end - fall_impact
        if imp_len > 0:
            imp_profile = np.sin(np.linspace(0, np.pi, imp_len))
            acc_z[fall_impact:imp_end] = -2.5 * imp_profile + 0.3
            acc_x[fall_impact:imp_end] = 4.2 * imp_profile
            acc_y[fall_impact:imp_end] = 1.8 * imp_profile
            gyr_z[fall_impact:imp_end] = 320.0 * imp_profile

        # 4. Post-impact / lying resting state
        if imp_end < n_samples:
            acc_x[imp_end:] = rng.normal(0.95, 0.02, n_samples - imp_end)  # lying flat
            acc_y[imp_end:] = rng.normal(0.05, 0.02, n_samples - imp_end)
            acc_z[imp_end:] = rng.normal(0.10, 0.02, n_samples - imp_end)
            gyr_x[imp_end:] = rng.normal(0.0, 1.5, n_samples - imp_end)
            gyr_y[imp_end:] = rng.normal(0.0, 1.5, n_samples - imp_end)
            gyr_z[imp_end:] = rng.normal(0.0, 1.5, n_samples - imp_end)

    df = pd.DataFrame({
        "TimeStamp(s)": np.round(time_stamp, 4),
        "FrameCounter": frame_counter,
        "AccX": np.round(acc_x, 4),
        "AccY": np.round(acc_y, 4),
        "AccZ": np.round(acc_z, 4),
        "GyrX": np.round(gyr_x, 2),
        "GyrY": np.round(gyr_y, 2),
        "GyrZ": np.round(gyr_z, 2),
        "EulerX": np.round(euler_x, 2),
        "EulerY": np.round(euler_y, 2),
        "EulerZ": np.round(euler_z, 2),
    })
    return df


def create_representative_dataset(
    sensor_dir: str = "data/kfall/sensor_data",
    label_dir: str = "data/kfall/label_data",
):
    """
    Creates representative test files across Train, Validation, and Test subjects.
    """
    os.makedirs(sensor_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    config = {
        "SA07": [
            ("T01", "R01", False, 0, 0, 350, 101),
            ("T02", "R01", False, 0, 0, 400, 102),
            ("T22", "R01", True, 150, 230, 500, 103),
            ("T26", "R01", True, 180, 250, 500, 104),
        ],
        "SA10": [
            ("T01", "R01", False, 0, 0, 350, 201),
            ("T22", "R01", True, 160, 240, 500, 202),
        ],
        "SA06": [
            ("T02", "R01", False, 0, 0, 400, 301),
            ("T26", "R01", True, 140, 220, 500, 302),
        ],
        "SA08": [
            ("T01", "R01", False, 0, 0, 350, 401),
            ("T23", "R01", True, 170, 245, 500, 402),
            ("T25", "R01", True, 160, 235, 500, 403),  # Intentionally missing from label sheet!
        ],
    }

    labels_by_subject = {
        "SA07": [
            {"Task Code": "T22", "Description": "Forward fall", "Trial ID": "R01", "Fall_onset_frame": 150, "Fall_Impact_frame": 230},
            {"Task Code": "T26", "Description": "Backward fall", "Trial ID": "R01", "Fall_onset_frame": 180, "Fall_Impact_frame": 250},
        ],
        "SA10": [
            {"Task Code": "T22", "Description": "Forward fall", "Trial ID": "R01", "Fall_onset_frame": 160, "Fall_Impact_frame": 240},
        ],
        "SA06": [
            {"Task Code": "T26", "Description": "Backward fall", "Trial ID": "R01", "Fall_onset_frame": 140, "Fall_Impact_frame": 220},
            {"Task Code": "T28", "Description": "Backward fall with knee flexed", "Trial ID": "R01", "Fall_onset_frame": 100, "Fall_Impact_frame": 180},  # No CSV file exists!
        ],
        "SA08": [
            {"Task Code": "T23", "Description": "Forward fall while jogging", "Trial ID": "R01", "Fall_onset_frame": 170, "Fall_Impact_frame": 245},
        ],
    }

    total_created = 0
    for sub, trials in config.items():
        sub_dir = os.path.join(sensor_dir, sub)
        os.makedirs(sub_dir, exist_ok=True)
        for task, trial, is_fall, onset, impact, n_samp, s in trials:
            df = generate_motion_recording(
                n_samples=n_samp,
                is_fall=is_fall,
                fall_onset=onset,
                fall_impact=impact,
                seed=s,
            )
            csv_path = os.path.join(sub_dir, f"{sub}{task}{trial}.csv")
            df.to_csv(csv_path, index=False)
            total_created += 1

    # Also add one corrupted/short file to verify quality audit rejection
    short_df = generate_motion_recording(n_samples=30, seed=999)
    short_path = os.path.join(sensor_dir, "SA08", "SA08T99R01.csv")
    short_df.to_csv(short_path, index=False)
    total_created += 1

    # Write label Excel files
    for sub, label_rows in labels_by_subject.items():
        ldf = pd.DataFrame(label_rows)
        excel_path = os.path.join(label_dir, f"{sub}_label.xlsx")
        ldf.to_excel(excel_path, index=False)

    print(f"Created {total_created} representative motion CSVs and {len(labels_by_subject)} label Excel files.")


if __name__ == "__main__":
    create_representative_dataset()
