// 烟雾测试：打开首页，看 console 错误，点 AI 按钮看 sidebar 是否打开，截图。
import puppeteer from 'puppeteer';
import { mkdir } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const outDir = resolve(__dirname, '..', 'screenshots', 'smoke');

async function main() {
  await mkdir(outDir, { recursive: true });
  const browser = await puppeteer.launch({ headless: 'new', args: ['--no-sandbox'] });
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900 });

  const errors = [];
  const consoleMsgs = [];
  page.on('console', (m) => consoleMsgs.push(`[${m.type()}] ${m.text()}`));
  page.on('pageerror', (e) => errors.push('PAGEERROR ' + e.message));
  page.on('requestfailed', (r) => errors.push('FAIL ' + r.url() + ' :: ' + (r.failure()?.errorText || '?')));

  await page.goto('http://localhost:4321/', { waitUntil: 'networkidle0', timeout: 30000 });
  await new Promise((r) => setTimeout(r, 1500));

  // initial screenshot
  await page.screenshot({ path: resolve(outDir, '01-loaded.png'), fullPage: false });

  // Click AI button
  const aiBtnExists = await page.$('#ai-open-btn');
  console.log('AI button found:', !!aiBtnExists);
  if (aiBtnExists) {
    await page.click('#ai-open-btn');
    await new Promise((r) => setTimeout(r, 600));
    const isOpen = await page.evaluate(() => {
      const s = document.getElementById('ai-sidebar');
      return s ? s.classList.contains('open') : 'no-sidebar';
    });
    console.log('AI sidebar open class:', isOpen);
    await page.screenshot({ path: resolve(outDir, '02-ai-open.png'), fullPage: false });
  }

  // Click Edit button
  const editBtnExists = await page.$('#edit-btn');
  console.log('Edit button found:', !!editBtnExists);
  if (editBtnExists) {
    await page.click('#edit-btn');
    await new Promise((r) => setTimeout(r, 600));
    const isEditorOpen = await page.evaluate(() => {
      const p = document.getElementById('editor-panel');
      return p ? p.classList.contains('open') : 'no-panel';
    });
    console.log('Editor panel open class:', isEditorOpen);
    await page.screenshot({ path: resolve(outDir, '03-edit-open.png'), fullPage: false });
  }

  await browser.close();

  console.log('\n=== console messages ===');
  consoleMsgs.forEach((m) => console.log(m));
  console.log('\n=== errors ===');
  if (!errors.length) console.log('  (none)');
  errors.forEach((e) => console.log(' ', e));
}

main().catch((e) => { console.error(e); process.exit(1); });
