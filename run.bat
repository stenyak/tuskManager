@echo off
rem Builds tuskManager if its sources changed, then launches it.
setlocal
cd /d "%~dp0"

rem The exe is copied here from the build files.
set "CARGO_TARGET_DIR=%~dp0target"

rem rustup adds cargo to PATH, but terminals opened before the install don't see it yet.
set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"

where cargo >nul 2>nul && set "HAVE_RUST=1"
rem Piping from an exe in a missing folder aborts the whole script, hence the "if exist".
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" "%VSWHERE%" -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | findstr . >nul && set "HAVE_BUILDTOOLS=1"
if "%HAVE_RUST%%HAVE_BUILDTOOLS%"=="11" goto :build

echo.
echo You need to install some tools:
echo  - Open cmd.exe
echo  - There, copypaste the command(s) below and press enter:
if not defined HAVE_BUILDTOOLS echo      winget install --id Microsoft.VisualStudio.2022.BuildTools -e --override "--passive --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
if not defined HAVE_RUST echo      winget install --id Rustlang.Rustup -e
echo  - Then press enter and rerun %~nx0 again
echo.
pause
exit /b 1

:build
echo Building tuskManager (the first build takes a few minutes)...
cargo build --release || (echo.&echo Build failed, see the errors above.&pause&exit /b 1)
copy /y "%CARGO_TARGET_DIR%\release\tuskManager.exe" tuskManager.exe >nul || (echo.&echo Close the running tuskManager first, so it can be updated.&pause&exit /b 1)
start "" tuskManager.exe
