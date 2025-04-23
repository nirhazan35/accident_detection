@echo off
echo Starting overnight training of all accident detection models
echo This will train 5 different architectures sequentially
echo Started at: %date% %time%
echo.

REM Activate virtual environment if using one
REM call path\to\venv\Scripts\activate.bat

REM Run the Python script
python run_all_models.py

echo.
echo All training completed at: %date% %time%
echo Results are saved in the logs directory
echo.

REM Keep the window open
pause 