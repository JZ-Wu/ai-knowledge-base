#!/bin/bash
cd "$(dirname "$0")"

if lsof -i :8001 &>/dev/null; then
    echo "Server already running on port 8001"
    exit 0
fi

nohup /usr/bin/python3 -m uvicorn server.main:app --host 0.0.0.0 --port 8001 > /tmp/knowledge-base.log 2>&1 &
echo "AI Knowledge Base Server started:"
echo "  Local:   http://localhost:8001/#/"
echo "  LAN:     http://192.168.1.108:8001/#/"
echo "Log: /tmp/knowledge-base.log"
