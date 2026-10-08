# Backup & Disaster Recovery Guide
## Standalone Wearable Fall Detection System

**Version**: Stage 5D Hardened Deployment  
**Project Path**: `C:\Users\ROHIT\Desktop\major project`

---

## 1. Critical Project Artifacts

To ensure complete disaster recovery, the following assets must be preserved:

| Component | Files / Directories | Importance |
| :--- | :--- | :--- |
| **Machine Learning Models** | `fall_detection_dataset/models/stage_4D/stage_4D_random_forest.joblib`<br>`fall_detection_dataset/models/stage_4D/scaler.joblib`<br>`models/selected_model.joblib`<br>`models/sisfall/sisfall_random_forest.joblib` | **CRITICAL** (Frozen trained weights) |
| **Feature Definitions** | `fall_detection_dataset/models/stage_4D/feature_order.json` | **CRITICAL** (Exact 100-feature mapping) |
| **Confirmation Config** | `fall_detection_dataset/models/stage_4E/confirmation_engine_config.json` | **CRITICAL** (Rule N3, threshold 0.60) |
| **Event Database** | `fall_detection_dataset/fall_events.db` | **CRITICAL** (Operational audit trail) |
| **Source Code** | `src/fall_event_backend/`<br>`src/fall_confirmation/` | **CRITICAL** (Application runtime) |
| **Web Dashboard** | `src/fall_event_backend/templates/index.html`<br>`src/fall_event_backend/static/` | **CRITICAL** (Monitoring interface) |
| **Validation Library** | `fall_detection_dataset/live_validation/stage_4G/` | **HIGH** (17 physical recording sessions) |
| **Verification Reports** | `fall_detection_dataset/reports/` | **HIGH** (Audit compliance evidence) |

---

## 2. SQLite Database Backup Procedures

Because the database runs in **Write-Ahead Logging (WAL)** mode, simple file copying while the application is actively writing could capture an incomplete WAL checkpoint. Follow either method below:

### Method A: Safe Online Backup via Python (Recommended)
You can take a safe, live backup without stopping the Flask backend:
```python
import sqlite3

src_path = "fall_detection_dataset/fall_events.db"
backup_path = "fall_detection_dataset/fall_events_backup.db"

with sqlite3.connect(src_path) as src_conn:
    with sqlite3.connect(backup_path) as dst_conn:
        src_conn.backup(dst_conn)

print(f"Hot backup completed successfully to: {backup_path}")
```

### Method B: Offline Cold File Backup
1. Stop the Flask server (`Ctrl+C` in terminal).
2. Ensure no lock files exist (`fall_events.db-wal` or `fall_events.db-shm` should be flushed).
3. Copy `fall_detection_dataset/fall_events.db` to your backup directory.

---

## 3. Database Restoration Procedure

If database corruption occurs or roll-back is required:
1. **Stop the Flask Server**: Terminate `run_fall_detection.bat` or the Python process.
2. **Quarantine Existing File**:
   ```cmd
   ren fall_detection_dataset\fall_events.db fall_events_corrupted.db
   ```
3. **Restore Backup File**:
   ```cmd
   copy fall_detection_dataset\fall_events_backup.db fall_detection_dataset\fall_events.db
   ```
4. **Restart Backend**:
   ```cmd
   run_fall_detection.bat
   ```
5. **Verify Health**: Visit `http://127.0.0.1:5000/api/health` and verify HTTP 200 response with `"status": "healthy"`.

---

## 4. Model Integrity Verification

To verify that model weights have not been corrupted or altered:
1. Run the hash verification command:
   ```cmd
   python -c "import hashlib; print('Model SHA-256:', hashlib.sha256(open('fall_detection_dataset/models/stage_4D/stage_4D_random_forest.joblib','rb').read()).hexdigest())"
   ```
2. Expected Reference Hash:
   `a499a0dd0107d9b25de01012a2efe8293f1b1e2092d1af51b08237efd3a8584c`
3. If the hash matches, the model is authentic and intact.
