"""启动知识库 AI 服务器。

用法:
    python run.py              # 默认 0.0.0.0:8001（局域网可访问）
    python run.py --port 8080  # 指定端口
    python run.py --host 127.0.0.1  # 仅本机访问
"""
import argparse
import socket
import uvicorn


def get_lan_ips():
    """尝试枚举本机所有非回环 IPv4 地址，方便手机接入。"""
    ips = []
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if ip and not ip.startswith("127.") and ip not in ips:
                ips.append(ip)
    except Exception:
        pass
    return ips


def main():
    parser = argparse.ArgumentParser(description="AI Knowledge Base Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    print(f"Starting server (binding {args.host}:{args.port})")
    print(f"  Local:   http://localhost:{args.port}/")
    if args.host == "0.0.0.0":
        for ip in get_lan_ips():
            print(f"  LAN:     http://{ip}:{args.port}/   (手机同 WiFi 可访问)")
    print("Press Ctrl+C to stop")

    uvicorn.run(
        "server.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
