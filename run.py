"""启动知识库 AI 服务器。

用法:
    python run.py              # 默认 127.0.0.1:8000
    python run.py --port 8080  # 指定端口
"""
import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="AI Knowledge Base Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    print(f"Starting server at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")

    uvicorn.run(
        "server.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
