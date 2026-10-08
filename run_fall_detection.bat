@echo off
setlocal
title Wearable Fall Detection System - Monitoring Server

echo ======================================================================
echo   STANDALONE WEARABLE FALL DETECTION SYSTEM
echo   Stage 5D Hardened Deployment Launcher
echo ======================================================================
echo.

:: Detect Python executable in PATH
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python was not found in your system PATH.
    echo Please ensure Python 3.10-3.13 is installed and added to PATH,
    echo or activate your Anaconda/venv environment before running this script.
    echo.
    pause
    exit /b 1
)

:: Display Python version
echo [INFO] Detected Python runtime:
python --version
echo.

:: Optional environment validation check
if exist "fall_detection_dataset\scripts\stage_5D\check_environment.py" (
    echo [INFO] Running quick environment diagnostic...
    python fall_detection_dataset\scripts\stage_5D\check_environment.py --quick
    if %ERRORLEVEL% NEQ 0 (
        echo [WARNING] Environment diagnostics reported warnings or missing dependencies.
        echo Review above messages or run 'pip install -r requirements.txt' if needed.
        echo.
    )
)

:: Production Configuration (Overridable via environment variables)
if "%SERIAL_PORT%"=="" set "SERIAL_PORT=COM22"
if "%SERIAL_BAUD%"=="" set "SERIAL_BAUD=460800"
if "%MQTT_BROKER_HOST%"=="" set "MQTT_BROKER_HOST=127.0.0.1"
if "%MQTT_BROKER_PORT%"=="" set "MQTT_BROKER_PORT=1883"
if "%BACKEND_HOST%"=="" set "BACKEND_HOST=127.0.0.1"
if "%BACKEND_PORT%"=="" set "BACKEND_PORT=5000"

echo [INFO] Starting Standalone Fall Detection Unified Backend & Gateway...
echo [INFO] Web Dashboard:  http://%BACKEND_HOST%:%BACKEND_PORT%/dashboard
echo [INFO] Service Health: http://%BACKEND_HOST%:%BACKEND_PORT%/api/health
echo [INFO] System Health:  http://%BACKEND_HOST%:%BACKEND_PORT%/api/system/health
echo [INFO] Hardware Port:  %SERIAL_PORT% @ %SERIAL_BAUD% baud
echo [INFO] MQTT Broker:    %MQTT_BROKER_HOST%:%MQTT_BROKER_PORT%
echo.
echo [NOTE] Security Boundary: Server binds strictly to %BACKEND_HOST% (Localhost only).
echo [NOTE] Press Ctrl+C in this terminal window to stop the server safely.
echo ======================================================================
echo.

:: Launch Stage 9 Production Launcher (orchestrating preflight, backend, gateway, and dashboard)
python scripts\stage_9\launcher.py %*

endlocal
