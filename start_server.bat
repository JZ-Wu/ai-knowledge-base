@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Starting AI Knowledge Base backend on http://localhost:8001
echo For the web UI on this port, build first: cd web && npm run build
python -m uvicorn server.main:app --host 0.0.0.0 --port 8001
pause
