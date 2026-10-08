"""
Temporal Windowing and Labeling Module for KFall.

Generates 50-sample sliding windows (0.50s @ 100 Hz, 50% overlap),
computes exact frame intervals, and labels windows based on ground-truth fall annotations.
"""

from typing import Dict, Generator, List, Optional, Tuple
import numpy as np
import pandas as pd


class WindowGenerator:
    """
    Slices a continuous motion recording into sliding temporal windows
    and attaches comprehensive fall-overlap metadata.
    """

    def __init__(self, window_size: int = 50, window_step: int = 25):
        self.window_size = window_size
        self.window_step = window_step

    def generate_windows(
        self,
        df: pd.DataFrame,
        is_fall_trial: bool,
        fall_onset_frame: Optional[int] = None,
        fall_impact_frame: Optional[int] = None,
    ) -> List[Dict]:
        """
        Slices a DataFrame into fixed-size windows and computes temporal labels.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned sensor recording containing FrameCounter and sensor signals.
        is_fall_trial : bool
            Whether the recording corresponds to a fall task (T22-T36).
        fall_onset_frame : int, optional
            Onset frame from the subject's label sheet (required if is_fall_trial).
        fall_impact_frame : int, optional
            Impact frame from the subject's label sheet (required if is_fall_trial).

        Returns
        -------
        list of dict, each dict containing window slice indices, metadata, and data array.
        """
        windows = []
        n_samples = len(df)
        if n_samples < self.window_size:
            return windows

        frame_counters = df["FrameCounter"].values
        timestamps = df["TimeStamp(s)"].values

        for start_idx in range(0, n_samples - self.window_size + 1, self.window_step):
            end_idx = start_idx + self.window_size - 1

            w_start_frame = int(frame_counters[start_idx])
            w_end_frame = int(frame_counters[end_idx])
            w_start_time = float(timestamps[start_idx])
            w_end_time = float(timestamps[end_idx])

            window_frames = frame_counters[start_idx : start_idx + self.window_size]

            # Labeling computation
            if not is_fall_trial or fall_onset_frame is None or fall_impact_frame is None:
                # ADL or missing fall label
                label = 0
                overlap_start = None
                overlap_end = None
                overlap_samples = 0
                overlap_ratio = 0.0
            else:
                # Fall trial with valid ground truth
                # Fall interval: [fall_onset_frame, fall_impact_frame]
                calc_overlap_start = max(w_start_frame, int(fall_onset_frame))
                calc_overlap_end = min(w_end_frame, int(fall_impact_frame))

                if calc_overlap_start <= calc_overlap_end:
                    overlap_start = int(calc_overlap_start)
                    overlap_end = int(calc_overlap_end)
                    # Count exact samples falling within the fall dynamic phase
                    in_fall_mask = (window_frames >= fall_onset_frame) & (
                        window_frames <= fall_impact_frame
                    )
                    overlap_samples = int(np.sum(in_fall_mask))
                    overlap_ratio = float(overlap_samples / self.window_size)
                    label = 1
                else:
                    overlap_start = None
                    overlap_end = None
                    overlap_samples = 0
                    overlap_ratio = 0.0
                    label = 0

            # Window data slice
            window_df = df.iloc[start_idx : start_idx + self.window_size]

            windows.append({
                "window_start_idx": start_idx,
                "window_end_idx": end_idx,
                "window_start_frame": w_start_frame,
                "window_end_frame": w_end_frame,
                "window_start_time": w_start_time,
                "window_end_time": w_end_time,
                "fall_onset_frame": int(fall_onset_frame) if fall_onset_frame is not None else None,
                "fall_impact_frame": int(fall_impact_frame) if fall_impact_frame is not None else None,
                "overlap_start": overlap_start,
                "overlap_end": overlap_end,
                "overlap_samples": overlap_samples,
                "overlap_ratio": overlap_ratio,
                "label": label,
                "window_df": window_df,
            })

        return windows
