@echo off
setlocal
cd /d "%~dp0"
title PCOS Clinical Decision Support

set "PYTHON="
where py >nul 2>nul && set "PYTHON=py"
if not defined PYTHON (
    where python >nul 2>nul && set "PYTHON=python"
)

if not defined PYTHON (
    echo.
    echo Python 3 could not be found on this computer.
    echo Please install Python 3 from https://www.python.org/downloads/ and try again.
    echo.
    pause
    exit /b 1
)

%PYTHON% -c "import flask, pandas, sklearn, shap, pyarrow, joblib" >nul 2>nul
if errorlevel 1 (
    echo.
    echo Preparing the application ^(first run only, this may take a few minutes^)...
    echo.
    %PYTHON% -m pip install --quiet --disable-pip-version-check -r requirements.txt
)

%PYTHON% -c "import flask, pandas, sklearn, shap, pyarrow, joblib" >nul 2>nul
if errorlevel 1 (
    echo.
    echo The application could not install its required components.
    echo Please check your internet connection and try again.
    echo.
    pause
    exit /b 1
)

%PYTHON% app.py
