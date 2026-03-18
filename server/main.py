from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from server.config import DOCS_ROOT
from server.routes import source, chat, edit, stats

app = FastAPI(title="AI Knowledge Base Server")

# API 路由
app.include_router(source.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(edit.router, prefix="/api")
app.include_router(stats.router, prefix="/api")

# 静态文件 (知识库目录) — 必须放在最后，作为 catch-all
app.mount("/", StaticFiles(directory=str(DOCS_ROOT), html=True), name="static")
