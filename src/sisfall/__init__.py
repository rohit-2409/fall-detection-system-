"""
SisFall Dataset Adapter and Parser Package.

Dedicated, standalone adapter for the SisFall fall detection dataset.
Provides strict parsing, physical unit conversion (ADXL345 to g, ITG3200 to deg/s),
zero-phase anti-aliasing decimation (200 Hz -> 100 Hz), and metadata extraction.

Safety Notice:
Existing KFall pipeline (Stage 1-3) is completely separate and frozen.
"""

from src.sisfall.sisfall_parser import (
    ADXL345_SCALE,
    ITG3200_SCALE,
    MMA8451Q_SCALE,
    BASELINE_COLUMNS,
    SisFallParseError,
    parse_sisfall_raw_file,
    parse_sisfall_line,
)
from src.sisfall.sisfall_loader import (
    SisFallMetadata,
    extract_sisfall_metadata,
    discover_sisfall_files,
    resample_200hz_to_100hz,
    load_sisfall_file,
)
from src.sisfall.sisfall_validation import (
    run_representative_validation,
    run_full_structural_validation,
)
from src.sisfall.sisfall_windowing import (
    SAMPLING_RATE_HZ,
    WINDOW_SIZE_SAMPLES,
    STEP_SIZE_SAMPLES,
    WINDOW_CHANNELS,
    compute_expected_window_count,
    generate_windows_from_trial,
    build_sisfall_window_dataset,
)
from src.sisfall.sisfall_window_validation import (
    run_representative_window_tests,
    validate_parquet_window_dataset,
)
from src.sisfall.sisfall_features import (
    ALL_SIGNALS,
    BASE_SIGNALS,
    DERIVED_SIGNALS,
    FEATURE_NAMES,
    METADATA_COLUMNS,
    STAT_NAMES,
    compute_signal_statistics_vectorized,
    extract_features_from_batch,
    build_sisfall_feature_dataset,
)
from src.sisfall.sisfall_feature_validation import (
    run_mathematical_reference_tests,
    validate_full_feature_dataset,
)

__all__ = [
    "ADXL345_SCALE",
    "ITG3200_SCALE",
    "MMA8451Q_SCALE",
    "BASELINE_COLUMNS",
    "SisFallParseError",
    "parse_sisfall_raw_file",
    "parse_sisfall_line",
    "SisFallMetadata",
    "extract_sisfall_metadata",
    "discover_sisfall_files",
    "resample_200hz_to_100hz",
    "load_sisfall_file",
    "run_representative_validation",
    "run_full_structural_validation",
    "SAMPLING_RATE_HZ",
    "WINDOW_SIZE_SAMPLES",
    "STEP_SIZE_SAMPLES",
    "WINDOW_CHANNELS",
    "compute_expected_window_count",
    "generate_windows_from_trial",
    "build_sisfall_window_dataset",
    "run_representative_window_tests",
    "validate_parquet_window_dataset",
    "ALL_SIGNALS",
    "BASE_SIGNALS",
    "DERIVED_SIGNALS",
    "FEATURE_NAMES",
    "METADATA_COLUMNS",
    "STAT_NAMES",
    "compute_signal_statistics_vectorized",
    "extract_features_from_batch",
    "build_sisfall_feature_dataset",
    "run_mathematical_reference_tests",
    "validate_full_feature_dataset",
]
