@echo off
echo ============================================================
echo  BetIQ — Calibrando parametros con datos historicos reales
echo  Fuente: football-data.co.uk (gratis, sin API key)
echo  Temporadas: 2022/23, 2023/24, 2024/25
echo  Ligas: Premier League, La Liga, Serie A, Bundesliga, Ligue 1
echo ============================================================
echo.
echo  Esto tarda ~2-5 minutos. Se ejecuta una vez (luego cada semana)
echo.
cd /d "%~dp0\backend"
python update_params.py

echo.
echo  Listo. Reinicia el backend para usar los nuevos parametros.
pause
