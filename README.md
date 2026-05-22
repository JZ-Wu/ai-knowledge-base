# AI 知识库

<div id="kb-stats" class="kb-stats"></div>

选中 Markdown 笔记里一段文字，让 AI 直接改这个文件。

<!-- ![demo](docs/assets/demo.gif) -->

## 安装

```bash
git clone https://github.com/JZ-Wu/ai-knowledge-base
cd ai-knowledge-base
pip install -r server/requirements.txt
python run.py            # http://localhost:8001
```

AI 后端二选一：

- `npm i -g @anthropic-ai/claude-code && claude` — 用 Claude 订阅，免 API key
- 或访问 `/docs/tools/settings.html` 填 OpenAI 兼容的 key（OpenAI、DeepSeek、Qwen 等）

第一次启动自动探测 `claude` 命令，找得到就用它，找不到就提示去填 key。

## 用法

打开任意页面，选中文字，点屏幕右下角浮动按钮，对 AI 说：

```
把这段公式推导补充完整
加一个 Switch Transformer vs Mixtral 的对比表格
把这章翻译成英文
```

AI 走 Read / Edit / Write 工具改对应的 `.md`，改完刷新就看到。工具白名单只放这五个，不会乱跑命令；改前自动 `.bak` 备份。

源码编辑器：`Ctrl+Shift+E`（CodeMirror，Markdown 高亮）。

## 多知识库

`knowledge_bases/<slug>/` 下每个子目录是一个独立 KB，URL `/kb/<slug>/`。顶栏 `KB ▾` 切换，设置页里新建 / 改名 / 上传 / 删除（删除是搬到 `_trash/`，不丢）。

自带 `ai-ml-interview`：AI/ML 面试笔记，166 篇 / 121 万字，覆盖大模型、机器学习基础、强化学习、视觉、具身智能、CUDA、分布式训练等。想用作自己的内容：删了，建个空 KB，拖文件夹上传。

## 技术栈

Docsify · FastAPI · Claude CLI / OpenAI 兼容 API · CodeMirror · KaTeX · PDF.js

架构、部署、EXTERNAL_MOUNTS、ACCESS_PASSWORD 等细节见 [INSTALL.md](INSTALL.md)。

## License

MIT
