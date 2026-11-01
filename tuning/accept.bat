@echo off
rem Keeps the current test-encoder.exe as baseline.exe, the build to compare
rem the next changes with. Run bench.bat first, so that it is up to date.
setlocal
cd /d "%~dp0"
if not exist test-encoder.exe goto missing
copy /y test-encoder.exe baseline.exe >nul || exit /b 1
echo baseline.exe updated
exit /b 0

:missing
echo accept.bat: no test-encoder.exe, run bench.bat first
exit /b 1
