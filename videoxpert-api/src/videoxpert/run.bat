@echo off
where dotnet >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo dotnet is not installed or not in PATH.
    exit /b 1
)

for /f "tokens=2 delims= " %%i in ('dotnet --version') do set DOTNET_VERSION=%%i
if not defined DOTNET_VERSION (
    for /f "delims=" %%i in ('dotnet --version') do set DOTNET_VERSION=%%i
)

set DOTNET_MAJOR=
for /f "tokens=1 delims=." %%a in ("%DOTNET_VERSION%") do set DOTNET_MAJOR=%%a

if not "%DOTNET_MAJOR%"=="9" (
    echo dotnet version 9 is required. Found version %DOTNET_VERSION%.
    exit /b 1
)

dotnet run -p:Platform=x64