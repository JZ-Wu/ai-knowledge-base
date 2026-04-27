@echo off
chcp 65001 >nul
cd /d "D:\OneDrive - Alanyhq Networks\私人知识库"
echo Starting Private Knowledge Base Server on http://localhost:8001
python -m uvicorn server.main:app --host 0.0.0.0 --port 8001
pause
