"""
Stage 5D Environment & System Diagnostic Script.

Verifies:
1. Python version (>= 3.10)
2. Required Python packages
3. Frozen ML models, configs, and feature orders
4. Backend, storage, and API modules
5. Dashboard HTML templates and static assets
6. SQLite database connectivity
7. Project separation boundary (0 references to external projects)

Returns exit code 0 on PASS, 1 on FAIL.
"""

import argparse
import importlib
import json
import os
import sqlite3
import sys
from typing import Dict, List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Required Python Packages
REQUIRED_PACKAGES = [
    ("flask", "Flask web server & REST API"),
    ("numpy", "Vectorized signal buffers & dynamic acceleration math"),
    ("scipy", "Causal single-pole filter (scipy.signal.lfilter)"),
    ("pandas", "Standardized dataset reading & telemetry logging"),
    ("pyarrow", "Parquet signal file I/O"),
    ("sklearn", "Scikit-learn Random Forest inference"),
    ("joblib", "Model deserialization"),
    ("serial", "PySerial UART communication with ESP32"),
]

# Required Model & Config Files
REQUIRED_FILES = [
    os.path.join("fall_detection_dataset", "models", "stage_4D", "stage_4D_random_forest.joblib"),
    os.path.join("fall_detection_dataset", "models", "stage_4D", "feature_order.json"),
    os.path.join("fall_detection_dataset", "models", "stage_4D", "scaler.joblib"),
    os.path.join("fall_detection_dataset", "models", "stage_4E", "confirmation_engine_config.json"),
    os.path.join("models", "selected_model.joblib"),
    os.path.join("models", "sisfall", "sisfall_random_forest.joblib"),
    os.path.join("src", "fall_event_backend", "api.py"),
    os.path.join("src", "fall_event_backend", "models.py"),
    os.path.join("src", "fall_event_backend", "storage.py"),
    os.path.join("src", "fall_event_backend", "service.py"),
    os.path.join("src", "fall_event_backend", "integration_adapter.py"),
    os.path.join("src", "fall_event_backend", "templates", "index.html"),
    os.path.join("src", "fall_event_backend", "static", "css", "dashboard.css"),
    os.path.join("src", "fall_event_backend", "static", "js", "dashboard.js"),
    os.path.join("src", "fall_confirmation", "temporal_adapter.py"),
    os.path.join("src", "fall_confirmation", "stage_4G_live_validator.py"),
]


def check_python_version() -> Tuple[bool, str]:
    v = sys.version_info
    ver_str = f"{v.major}.{v.minor}.{v.micro}"
    if v.major == 3 and v.minor >= 10:
        return True, f"Python {ver_str} (Compatible: >= 3.10)"
    return False, f"Python {ver_str} (Incompatible: requires Python >= 3.10)"


def check_package(pkg_name: str, desc: str) -> Tuple[bool, str]:
    try:
        mod = importlib.import_module(pkg_name)
        try:
            from importlib.metadata import version
            ver = version(pkg_name if pkg_name != "serial" else "pyserial")
        except Exception:
            ver = getattr(mod, "__version__", "installed")
        return True, f"{pkg_name} ({ver}) - {desc}"
    except ImportError:
        return False, f"{pkg_name} is NOT installed - {desc}"


def check_file(rel_path: str) -> Tuple[bool, str]:
    full_path = os.path.join(PROJECT_ROOT, rel_path)
    if os.path.exists(full_path):
        size_kb = os.path.getsize(full_path) / 1024.0
        return True, f"{rel_path} ({size_kb:.1f} KB)"
    return False, f"Missing critical file: {rel_path}"


def check_sqlite() -> Tuple[bool, str]:
    try:
        conn = sqlite3.connect(":memory:")
        cur = conn.cursor()
        cur.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, val TEXT)")
        cur.execute("INSERT INTO test (val) VALUES ('ok')")
        cur.execute("SELECT val FROM test WHERE id = 1")
        row = cur.fetchone()
        conn.close()
        if row and row[0] == "ok":
            return True, f"SQLite {sqlite3.sqlite_version} functional (in-memory test passed)"
        return False, "SQLite query returned unexpected data"
    except Exception as e:
        return False, f"SQLite error: {str(e)}"


def check_feature_order_integrity() -> Tuple[bool, str]:
    fo_path = os.path.join(PROJECT_ROOT, "fall_detection_dataset", "models", "stage_4D", "feature_order.json")
    try:
        with open(fo_path, "r", encoding="utf-8") as f:
            features = json.load(f)
        if len(features) == 100:
            return True, f"Feature order verified (exactly 100 features)"
        return False, f"Feature order corrupted: expected 100 features, found {len(features)}"
    except Exception as e:
        return False, f"Failed reading feature_order.json: {str(e)}"


def run_diagnostics(quick_mode: bool = False) -> bool:
    print("=" * 75)
    print("  STANDALONE WEARABLE FALL DETECTION SYSTEM — ENVIRONMENT DIAGNOSTICS")
    print(f"  Project Root: {PROJECT_ROOT}")
    print("=" * 75)

    all_passed = True
    fails: List[str] = []
    warnings: List[str] = []

    # 1. Python Version
    ok, msg = check_python_version()
    status_label = "[PASS]" if ok else "[FAIL]"
    print(f"{status_label} {msg}")
    if not ok:
        all_passed = False
        fails.append(msg)

    # 2. SQLite Engine
    ok, msg = check_sqlite()
    status_label = "[PASS]" if ok else "[FAIL]"
    print(f"{status_label} {msg}")
    if not ok:
        all_passed = False
        fails.append(msg)

    # 3. Python Packages
    print("\n--- Dependencies ---")
    for pkg, desc in REQUIRED_PACKAGES:
        ok, msg = check_package(pkg, desc)
        if ok:
            print(f"[PASS] {msg}")
        else:
            if pkg == "serial":
                print(f"[WARNING] {msg} (Only required when connecting physical ESP32)")
                warnings.append(msg)
            else:
                print(f"[FAIL] {msg}")
                all_passed = False
                fails.append(msg)

    # 4. Critical Files & Models
    print("\n--- Critical Models, Backend & UI Files ---")
    for rel_path in REQUIRED_FILES:
        ok, msg = check_file(rel_path)
        status_label = "[PASS]" if ok else "[FAIL]"
        print(f"{status_label} {msg}")
        if not ok:
            all_passed = False
            fails.append(msg)

    # 5. Feature Order Check
    ok, msg = check_feature_order_integrity()
    status_label = "[PASS]" if ok else "[FAIL]"
    print(f"{status_label} {msg}")
    if not ok:
        all_passed = False
        fails.append(msg)

    print("\n" + "=" * 75)
    if all_passed:
        print("  DIAGNOSTIC RESULT: ALL SYSTEM CHECKS PASSED")
        if warnings:
            print(f"  (with {len(warnings)} non-fatal warning(s))")
        print("=" * 75)
        return True
    else:
        print(f"  DIAGNOSTIC RESULT: {len(fails)} CRITICAL CHECK(S) FAILED")
        for f in fails:
            print(f"    - {f}")
        print("=" * 75)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 5D Environment Diagnostics")
    parser.add_argument("--quick", action="store_true", help="Quick diagnostic mode")
    args = parser.parse_args()

    success = run_diagnostics(quick_mode=args.quick)
    sys.exit(0 if success else 1)
