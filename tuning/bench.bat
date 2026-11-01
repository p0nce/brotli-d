@echo off
rem Builds test-encoder for native ARM64 and runs it.
rem   bench.bat               levels 0 to 11
rem   bench.bat 5             level 5
rem   bench.bat 4 9           levels 4 to 9
rem No ( ) blocks: %ProgramFiles(x86)% has a ")" that would close them.
setlocal
cd /d "%~dp0"

rem The ARM64 linker and libraries come from the MSVC ARM64 environment.
if /i "%VSCMD_ARG_TGT_ARCH%"=="arm64" goto build
rem On PATH, so that the "(x86)" path stays out of the for command.
set "PATH=%PATH%;%ProgramFiles(x86)%\Microsoft Visual Studio\Installer"
for /f "usebackq delims=" %%i in (`vswhere -latest -products * -property installationPath`) do set "VSDIR=%%i"
if not defined VSDIR goto novs
call "%VSDIR%\VC\Auxiliary\Build\vcvarsarm64.bat" >nul || exit /b 1
cd /d "%~dp0"

:build
dub build --compiler=ldc2 -b release-nobounds --combined --arch=aarch64-windows-msvc || exit /b 1

if "%~1"=="" goto all
"%~dp0test-encoder.exe" %*
exit /b %errorlevel%

:all
"%~dp0test-encoder.exe" 0 11
exit /b %errorlevel%

:novs
echo bench.bat: Visual Studio not found
exit /b 1
