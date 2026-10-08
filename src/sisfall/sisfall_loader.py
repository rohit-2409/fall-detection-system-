"""
SisFall Dataset Loader, Metadata Extraction, and Controlled Resampling Module.

Provides discovery of SisFall recordings, extraction of trial metadata with parent-folder
subject resolution (correcting the SA15 quirk), controlled 2:1 anti-aliasing decimation
from 200 Hz to 100 Hz using zero-phase Chebyshev IIR filtering, and conversion to DataFrames.
"""

import os
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.sisfall.sisfall_parser import (
    BASELINE_COLUMNS,
    SisFallParseError,
    parse_sisfall_raw_file,
)

# Optional SciPy import for anti-aliasing low-pass decimation
try:
    import scipy.signal as signal
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


@dataclass(frozen=True)
class SisFallMetadata:
    """Immutable metadata container for a SisFall trial recording."""
    subject_id: str
    activity_id: str
    activity_type: str  # 'ADL' or 'FALL'
    trial_id: str
    source_file: str
    source_path: str
    original_samples: int
    original_fs_hz: float
    resampled_samples: int
    resampled_fs_hz: float
    decimation_factor: int
    filter_method: str

    def to_dict(self) -> Dict[str, Union[str, int, float]]:
        return asdict(self)


def extract_sisfall_metadata(file_path: str) -> Dict[str, str]:
    """
    Extracts metadata from a SisFall recording path.

    CRITICAL RULE (Section 8):
    The subject identity MUST be derived from the immediate parent directory name,
    NOT solely from the filename. This guarantees that files such as:
        SA15/D17_SE15_R01.txt
    are correctly resolved to subject_id == 'SA15'.

    Parameters
    ----------
    file_path : str
        Path to the SisFall .txt file.

    Returns
    -------
    Dict[str, str]
        Dictionary with keys: subject_id, activity_id, activity_type, trial_id,
        source_file, source_path.

    Raises
    ------
    ValueError
        If the filename or directory structure does not conform to SisFall naming conventions.
    """
    norm_path = os.path.normpath(file_path)
    filename = os.path.basename(norm_path)
    parent_dir = os.path.basename(os.path.dirname(norm_path))

    # Validate parent directory as subject identifier (SA01..SA23, SE01..SE15)
    subj_match = re.match(r"^(S[AE]\d{2})$", parent_dir, re.IGNORECASE)
    if not subj_match:
        raise ValueError(
            f"Parent directory '{parent_dir}' of file '{filename}' is not a valid SisFall subject ID (expected SAxx or SExx)."
        )
    subject_id = subj_match.group(1).upper()

    # Parse filename components: <Activity>_<SubjectInFilename>_<Trial>.txt
    # Example: D01_SA01_R01.txt, F05_SA12_R04.txt, D17_SE15_R01.txt (inside SA15)
    file_match = re.match(r"^([DF]\d{2})_([A-Z0-9]+)_R(\d{2})\.txt$", filename, re.IGNORECASE)
    if not file_match:
        raise ValueError(
            f"Filename '{filename}' does not follow SisFall pattern '<Activity>_<Subject>_R<Trial>.txt'."
        )

    act_code, raw_subj_in_fn, trial_num = file_match.groups()
    activity_id = act_code.upper()
    trial_id = f"R{int(trial_num):02d}"

    # Determine activity type
    if activity_id.startswith("D"):
        activity_type = "ADL"
    elif activity_id.startswith("F"):
        activity_type = "FALL"
    else:
        raise ValueError(f"Unrecognized activity code '{activity_id}' in {filename}.")

    return {
        "subject_id": subject_id,
        "activity_id": activity_id,
        "activity_type": activity_type,
        "trial_id": trial_id,
        "source_file": filename,
        "source_path": norm_path,
        "filename_subject": raw_subj_in_fn.upper(),
    }


def discover_sisfall_files(
    dataset_root: str = r"data/SisFall_dataset/SisFall_dataset",
) -> List[str]:
    """
    Discovers all valid SisFall .txt recording paths within the dataset root.

    Strictly ignores:
    - '__MACOSX' directories
    - 'SA04/images/' or any non-subject subfolders
    - 'desktop.ini', '.DS_Store', '.png', and any non-.txt files

    Parameters
    ----------
    dataset_root : str
        Path to the SisFall dataset root directory.

    Returns
    -------
    List[str]
        Alphabetically sorted list of valid .txt file paths.
    """
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"SisFall dataset root not found: {dataset_root}")

    valid_files: List[str] = []

    # Enumerate subject folders
    for entry in sorted(os.listdir(dataset_root)):
        subj_dir = os.path.join(dataset_root, entry)
        if not os.path.isdir(subj_dir):
            continue

        # Skip non-subject folders such as __MACOSX
        if not re.match(r"^S[AE]\d{2}$", entry, re.IGNORECASE):
            continue

        for fname in sorted(os.listdir(subj_dir)):
            if not fname.lower().endswith(".txt"):
                continue

            full_path = os.path.join(subj_dir, fname)
            if os.path.isfile(full_path):
                valid_files.append(os.path.normpath(full_path))

    return valid_files


def resample_200hz_to_100hz(
    data: np.ndarray,
    q: int = 2,
) -> np.ndarray:
    """
    Downsamples a 2D sensor array from 200 Hz to 100 Hz (q=2) using a controlled
    anti-aliasing low-pass filter followed by decimation.

    Filter Specification:
    - Method: scipy.signal.decimate with ftype='iir', zero_phase=True
    - Filter Topology: 8th-order Chebyshev Type I lowpass filter
    - Cutoff: 0.8 * (fs / 2) / q = 40 Hz (prevents aliasing into the 50 Hz Nyquist band)
    - Passband Ripple: 0.05 dB
    - Phase Response: Exactly zero-phase via bidirectional filtfilt (no peak shift / phase distortion)

    Parameters
    ----------
    data : np.ndarray, shape (N, C)
        Input sensor time-series sampled at 200 Hz.
    q : int, default=2
        Decimation factor.

    Returns
    -------
    np.ndarray, shape (ceil(N/q), C)
        Resampled sensor time-series at 100 Hz.

    Raises
    ------
    ImportError
        If scipy is not installed in the active environment.
    """
    if not _SCIPY_AVAILABLE:
        raise ImportError(
            "SciPy is required for controlled anti-aliasing resampling (scipy.signal.decimate). "
            "Please ensure SciPy is installed in your scientific Python environment."
        )

    if q == 1:
        return data.copy()

    # Apply zero-phase 8th-order Chebyshev Type I decimation along time axis (axis=0)
    resampled = signal.decimate(data, q=q, ftype="iir", axis=0, zero_phase=True)
    return resampled.astype(np.float64)


def load_sisfall_file(
    file_path: str,
    resample_to_100hz: bool = True,
) -> Tuple[pd.DataFrame, SisFallMetadata]:
    """
    End-to-end loader for a single SisFall recording.

    Steps:
    1. Extracts trial metadata (subject_id from parent folder).
    2. Parses raw 9-column ASCII records and verifies format.
    3. Converts ADXL345 (to g) and ITG3200 (to deg/s).
    4. Applies zero-phase anti-aliasing decimation (200 Hz -> 100 Hz) if requested.
    5. Packages data into a clean pandas DataFrame with physical unit columns and Time_s.
    6. Returns (DataFrame, SisFallMetadata).

    Parameters
    ----------
    file_path : str
        Path to the SisFall .txt file.
    resample_to_100hz : bool, default=True
        Whether to resample from 200 Hz to 100 Hz.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns:
        ['Time_s', 'AccX_g', 'AccY_g', 'AccZ_g', 'GyrX_dps', 'GyrY_dps', 'GyrZ_dps']
    meta : SisFallMetadata
        Immutable metadata object.
    """
    meta_dict = extract_sisfall_metadata(file_path)
    raw_counts, converted_data, channel_names = parse_sisfall_raw_file(
        file_path, include_mma8451q=False
    )

    original_samples = converted_data.shape[0]
    original_fs_hz = 200.0

    if resample_to_100hz:
        resampled_data = resample_200hz_to_100hz(converted_data, q=2)
        resampled_samples = resampled_data.shape[0]
        resampled_fs_hz = 100.0
        decimation_factor = 2
        filter_method = (
            "scipy.signal.decimate(q=2, ftype='iir', zero_phase=True) "
            "[8th-order Chebyshev Type I, 0.05 dB ripple, 40 Hz cutoff, bidirectional filtfilt]"
        )
        dt = 1.0 / resampled_fs_hz
        time_axis = np.arange(resampled_samples, dtype=np.float64) * dt
        data_to_df = resampled_data
    else:
        resampled_samples = original_samples
        resampled_fs_hz = 200.0
        decimation_factor = 1
        filter_method = "None (Original 200 Hz)"
        dt = 1.0 / original_fs_hz
        time_axis = np.arange(original_samples, dtype=np.float64) * dt
        data_to_df = converted_data

    # Construct DataFrame
    df = pd.DataFrame(data_to_df, columns=channel_names)
    df.insert(0, "Time_s", time_axis)

    meta = SisFallMetadata(
        subject_id=meta_dict["subject_id"],
        activity_id=meta_dict["activity_id"],
        activity_type=meta_dict["activity_type"],
        trial_id=meta_dict["trial_id"],
        source_file=meta_dict["source_file"],
        source_path=meta_dict["source_path"],
        original_samples=original_samples,
        original_fs_hz=original_fs_hz,
        resampled_samples=resampled_samples,
        resampled_fs_hz=resampled_fs_hz,
        decimation_factor=decimation_factor,
        filter_method=filter_method,
    )

    return df, meta
