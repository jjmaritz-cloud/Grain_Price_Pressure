@echo off
title El Nino + Grain Price Pressure App

echo =========================================
echo  El Nino + Grain Price Pressure App
echo =========================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found.
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure you tick "Add Python to PATH" during installation.
    pause
    exit /b
)

echo Checking required Python packages...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

:restart
cls
echo =========================================
echo  El Nino + Grain Price Pressure App
echo =========================================
echo.
echo Starting Streamlit app...
echo.
echo When you want to stop the app, press CTRL + C.
echo After it stops, this window will stay open.
echo Update your code, then press any key here to restart.
echo.

python -m streamlit run streamlit_app.py

echo.
echo =========================================
echo  Streamlit app has stopped.
echo =========================================
echo.
echo Make your code changes now.
echo Then press any key to restart the app.
echo Or close this window to exit.
echo.
pause >nul

goto restart
