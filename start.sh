#!/bin/bash
cd "$(dirname "$0")"

if lsof -i :8000 &>/dev/null; then
    echo "Server already running on port 8000"
    exit 0
fi

nohup /usr/bin/python3 -m uvicorn server.main:app --host 0.0.0.0 --port 8000 > /tmp/knowledge-base.log 2>&1 &
echo "AI Knowledge Base Server started:"
echo "  Local:   http://localhost:8000/#/"
echo "  LAN:     http://192.168.1.108:8000/#/"
echo "Log: /tmp/knowledge-base.log"
