@echo off
REM SENTINEL stage runner (cmd.exe wrapper around run.ps1).
REM Usage:  run.bat verify-data        run.bat phase1        run.bat to-parquet --mode dev
setlocal
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "PYTHONUTF8=1"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
endlocal
