@echo off
echo ============================================================
echo  BetIQ Dashboard — Iniciando Backend
echo ============================================================
echo.
echo  Iniciando servidor en http://localhost:8000
echo  Presiona Ctrl+C para detener
echo.

:: Configura tu API key de https://the-odds-api.com
:: Obtén una gratis en: https://the-odds-api.com
set ODDS_API_KEY=TU_API_KEY_AQUI

cd /d "%~dp0\backend"
python -m uvicorn main:app --reload --port 8000

pause
