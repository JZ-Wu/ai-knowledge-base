import puppeteer from 'puppeteer';
import { mkdir } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
const __dirname = dirname(fileURLToPath(import.meta.url));
const outDir = resolve(__dirname, '..', 'screenshots', 'real');
await mkdir(outDir, { recursive: true });

// 2026-05-29 起改多 KB 布局：/kb/<slug>/...
const urls = [
  ['home', '/'],
  ['kb-interactive-dt',     '/kb/interactive-dt/'],
  ['kb-surgical-vqla',      '/kb/surgical-vqla-agent/'],
  ['kb-idea-research',      '/kb/idea-research/'],
  ['kb-chemistry',          '/kb/chemistry/'],
  ['paper-artgs',           '/kb/interactive-dt/wiki/papers/artgs/'],
  ['paper-surgical-vqla',   '/kb/surgical-vqla-agent/wiki/papers/surgical-vqla/'],
  ['methods-paper-reading', '/kb/idea-research/methods/paper-critical-reading/'],
];
const browser = await puppeteer.launch({ headless: 'new', args: ['--no-sandbox'] });
for (const [name, url] of urls) {
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900 });
  await page.goto('http://localhost:4321' + url, { waitUntil: 'networkidle0', timeout: 30000 });
  await new Promise(r => setTimeout(r, 1200));
  await page.screenshot({ path: resolve(outDir, name + '.png'), fullPage: false });
  console.log('saved', name);
  await page.close();
}
await browser.close();
