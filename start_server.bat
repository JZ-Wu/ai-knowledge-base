@echo off
chcp 65001 >nul
cd /d "D:\OneDrive - Alanyhq Networks\知识库"
echo Starting AI Knowledge Base Server on http://localhost:8000
python -m uvicorn server.main:app --host 0.0.0.0 --port 8000
pause
