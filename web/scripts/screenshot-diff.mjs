// 拍照对比脚本：demo（preview）vs Astro 重建版。
// 用法：node scripts/screenshot-diff.mjs <iter-number>
// 写出到 web/screenshots/iter-N/
// 需要：Astro :4321 + FastAPI :8001 都在跑

import puppeteer from 'puppeteer';
import { mkdir } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const iter = process.argv[2] || 'latest';
const outDir = resolve(__dirname, '..', 'screenshots', `iter-${iter}`);

const VIEWPORT = { width: 1440, height: 900 };

const TARGETS = [
  { name: 'demo-home',    url: 'http://localhost:8001/docs/preview/',  isDemo: true  },
  { name: 'astro-home',   url: 'http://localhost:4321/',               isDemo: false },
  { name: 'astro-topic',  url: 'http://localhost:4321/topics/interactive-dt/', isDemo: false },
  { name: 'astro-paper',  url: 'http://localhost:4321/topics/interactive-dt/artgs/', isDemo: false },
];

async function captureOne(browser, target, mode) {
  const page = await browser.newPage();
  await page.setViewport(VIEWPORT);

  // first go to the origin so we can set localStorage there
  await page.goto(target.url, { waitUntil: 'domcontentloaded', timeout: 30000 });

  await page.evaluate(({ mode, isDemo }) => {
    try {
      if (isDemo) {
        localStorage.setItem('preview_theme', 'warm-material');
        localStorage.setItem('preview_mode', mode);
      } else {
        localStorage.setItem('kb_mode', mode);
      }
    } catch (_) {}
  }, { mode, isDemo: target.isDemo });

  // reload to apply
  await page.goto(target.url, { waitUntil: 'networkidle0', timeout: 30000 });
  await new Promise((r) => setTimeout(r, 800)); // wait for stats fetch, TOC build

  const path = resolve(outDir, `${target.name}--${mode}.png`);
  await page.screenshot({ path, fullPage: true });
  console.log('  saved', path);
  await page.close();
}

async function main() {
  await mkdir(outDir, { recursive: true });
  const browser = await puppeteer.launch({
    headless: 'new',
    defaultViewport: VIEWPORT,
    args: ['--no-sandbox'],
  });
  try {
    for (const t of TARGETS) {
      for (const mode of ['light', 'dark']) {
        await captureOne(browser, t, mode);
      }
    }
  } finally {
    await browser.close();
  }
  console.log(`done → ${outDir}`);
}

main().catch((e) => { console.error(e); process.exit(1); });
