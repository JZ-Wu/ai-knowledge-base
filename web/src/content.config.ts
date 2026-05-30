// Content collection 配置：把 KB 的 markdown 拉进 Astro 内容层。
//
// 数据流：
//   原始 markdown 在 KB 根（../idea-research/、../普通化学/、../科研方法论/ 等）
//   ← scripts/sync-content.mjs 复制到 web/src/content/docs/...
//   ← Astro 内容集合 loader 读取
//   ← pages/[...slug].astro 渲染
//
// 这套设计让 Docsify（直接读 KB 根）和 Astro（读 src/content）同时可用，
// 内容只在迁移时同步一次。后续可写一个 watch 脚本自动同步。

import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const docs = defineCollection({
  loader: glob({
    pattern: '**/*.md',
    base: './src/content/docs',
    // 默认 glob 用 github-slugger 把 id 全转小写（EIT东方理工 → eit东方理工），
    // 但 KB 里的内部链接和文件名保留原始大小写 → 在大小写敏感的文件系统（Linux 部署）
    // 上全部 404。这里覆盖成"路径原样去扩展名"，让 URL == 文件路径，
    // 跟 CLAUDE.md 的"绝对路径 = 文件路径"约定一致。
    generateId: ({ entry }) => entry.replace(/\.(md|mdx)$/i, ''),
  }),
  schema: z.object({
    title: z.string().optional(),
    description: z.string().optional(),
    // frontmatter 可选，markdown 里大多数没有
  }),
});

export const collections = { docs };
