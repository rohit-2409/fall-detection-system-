"""
SisFall Raw Data Parser and Unit Conversion Module.

Handles strict parsing of SisFall 9-column semicolon-terminated ASCII text records,
channel extraction for the baseline 6-DOF sensor configuration (ADXL345 + ITG3200),
and exact physical unit conversion into gravity (g) and angular velocity (deg/s).
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np

# Official SisFall Conversion Factors (from SisFall Readme.txt)
# ADXL345: 13-bit resolution, +-16g range
# Equation: [(2 * Range) / (2^Resolution)] * AD = (2 * 16) / (2^13) = 32 / 8192 = 0.00390625 g / LSB
ADXL345_SCALE: float = 0.00390625

# ITG3200: 16-bit resolution, +-2000 deg/s range
# Equation: [(2 * Range) / (2^Resolution)] * RD = (2 * 2000) / (2^16) = 4000 / 65536 = 0.06103515625 deg/s / LSB
ITG3200_SCALE: float = 0.06103515625

# MMA8451Q: 14-bit resolution, +-8g range (documented for reference, excluded from baseline)
# Equation: [(2 * Range) / (2^Resolution)] * AD = (2 * 8) / (2^14) = 16 / 16384 = 0.0009765625 g / LSB
MMA8451Q_SCALE: float = 0.0009765625

# Baseline 6-channel column names matching project physical units
BASELINE_COLUMNS: List[str] = [
    "AccX_g",
    "AccY_g",
    "AccZ_g",
    "GyrX_dps",
    "GyrY_dps",
    "GyrZ_dps",
]


class SisFallParseError(ValueError):
    """Exception raised when a SisFall raw data file or row fails schema validation."""
    pass


def parse_sisfall_line(line: str, line_no: int, filename: str = "") -> List[int]:
    """
    Parses and validates a single line of a SisFall data file.

    Parameters
    ----------
    line : str
        Raw string line from file.
    line_no : int
        1-based line number for error reporting.
    filename : str
        Filename for context in error messages.

    Returns
    -------
    List[int]
        List of 9 signed integer sensor values.

    Raises
    ------
    SisFallParseError
        If line violates format (missing terminal semicolon, not 9 fields, non-integer).
    """
    stripped = line.strip()
    if not stripped:
        raise SisFallParseError(
            f"Empty line encountered at {filename}:{line_no}."
        )

    if not stripped.endswith(";"):
        raise SisFallParseError(
            f"Missing terminal semicolon at {filename}:{line_no}. Content: '{stripped[:40]}'"
        )

    # Strip semicolon and trailing whitespace
    body = stripped[:-1].strip()
    fields = body.split(",")

    if len(fields) != 9:
        raise SisFallParseError(
            f"Invalid column count at {filename}:{line_no}. Expected exactly 9 fields, got {len(fields)}. Content: '{stripped[:40]}'"
        )

    values: List[int] = []
    for col_idx, field in enumerate(fields):
        trimmed = field.strip()
        try:
            val = int(trimmed)
            values.append(val)
        except ValueError as err:
            raise SisFallParseError(
                f"Non-integer value '{trimmed}' in column {col_idx + 1} at {filename}:{line_no}."
            ) from err

    return values


def parse_sisfall_raw_file(
    file_path: str,
    include_mma8451q: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Parses an entire SisFall .txt file, validates schema, and converts baseline channels.

    Parameters
    ----------
    file_path : str
        Path to the SisFall .txt file.
    include_mma8451q : bool, default=False
        If True, also returns MMA8451Q columns. Default is False (baseline 6 channels).

    Returns
    -------
    raw_counts : np.ndarray, shape (N, 6) or (N, 9), dtype=int32
        Raw integer counts extracted from the file.
    converted_data : np.ndarray, shape (N, 6) or (N, 9), dtype=float64
        Data converted to physical units (g and deg/s).
    column_names : List[str]
        Names of the returned channels.

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    SisFallParseError
        If file is empty, corrupt, or violates SisFall formatting rules.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"SisFall file not found: {file_path}")

    filename = os.path.basename(file_path)
    all_raw_rows: List[List[int]] = []

    with open(file_path, "r", encoding="utf-8", errors="strict") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row_vals = parse_sisfall_line(stripped, line_no=line_no, filename=filename)
            all_raw_rows.append(row_vals)

    if not all_raw_rows:
        raise SisFallParseError(f"SisFall file {filename} contains zero valid data rows.")

    # Convert to 2D numpy array: shape (N, 9)
    raw_9col = np.array(all_raw_rows, dtype=np.int32)

    # Baseline channels:
    # Cols 0..2: ADXL345 (AccX, AccY, AccZ)
    # Cols 3..5: ITG3200 (GyrX, GyrY, GyrZ)
    raw_baseline = raw_9col[:, 0:6]

    # Convert to physical units
    converted_baseline = np.empty_like(raw_baseline, dtype=np.float64)
    # ADXL345 -> g
    converted_baseline[:, 0:3] = raw_baseline[:, 0:3].astype(np.float64) * ADXL345_SCALE
    # ITG3200 -> deg/s
    converted_baseline[:, 3:6] = raw_baseline[:, 3:6].astype(np.float64) * ITG3200_SCALE

    # Verify no NaN or Inf introduced
    if np.isnan(converted_baseline).any() or np.isinf(converted_baseline).any():
        raise SisFallParseError(
            f"Physical unit conversion resulted in NaN or Inf in {filename}."
        )

    if include_mma8451q:
        raw_mma = raw_9col[:, 6:9]
        converted_mma = raw_mma.astype(np.float64) * MMA8451Q_SCALE
        raw_out = raw_9col
        conv_out = np.hstack([converted_baseline, converted_mma])
        cols = BASELINE_COLUMNS + ["MMA_AccX_g", "MMA_AccY_g", "MMA_AccZ_g"]
        return raw_out, conv_out, cols

    return raw_baseline, converted_baseline, BASELINE_COLUMNS
