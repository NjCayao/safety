@echo off
cd /d "C:\xampp\htdocs\safety_system\operators"
echo [%date% %time%] Iniciando calibracion automatica para DNI 47469940 >> "C:\xampp\htdocs\safety_system\operators\logs\auto_calibration_47469940_20250712032623.log"
python calibrate_single_operator.py "47469940" >> "C:\xampp\htdocs\safety_system\operators\logs\auto_calibration_47469940_20250712032623.log" 2>&1
echo [%date% %time%] Calibracion finalizada con codigo %errorlevel% >> "C:\xampp\htdocs\safety_system\operators\logs\auto_calibration_47469940_20250712032623.log"
