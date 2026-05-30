// @ts-check
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// 自家从零重写的 KB 前端（无 Starlight 依赖）
// 视觉目标：完全对齐 /docs/preview/ 里的 ★ Warm·M demo
export default defineConfig({
  server: { port: 4321, host: true },

  // URL 末尾统一带 /，跟 KB markdown 里 [link](/foo/bar.md) 解析一致
  trailingSlash: 'always',

  // Dev toolbar 在截图里挡内容，关掉
  devToolbar: { enabled: false },

  // Dev: vite 代理 /api/* 到 FastAPI :8001
  vite: {
    server: {
      proxy: {
        '/api': {
          target: 'http://localhost:8001',
          changeOrigin: true,
        },
      },
    },
  },

  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
    // 用 shiki 主题做语法高亮（Astro 内置）
    shikiConfig: {
      themes: { light: 'github-light', dark: 'one-dark-pro' },
      defaultColor: false, // 让我的 CSS 通过 [data-theme] 切换
      wrap: false,
    },
  },

  integrations: [mdx()],
});
