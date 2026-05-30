import puppeteer from 'puppeteer';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
const __dirname = dirname(fileURLToPath(import.meta.url));
const errors = [];
const browser = await puppeteer.launch({ headless: 'new', args: ['--no-sandbox'] });
const page = await browser.newPage();
page.on('pageerror', (e) => errors.push('ERR ' + e.message));
page.on('requestfailed', (r) => errors.push('FAIL ' + r.url() + ' :: ' + (r.failure()?.errorText || '?')));
await page.setViewport({ width: 1440, height: 900 });
console.log('→ Home');
await page.goto('http://localhost:4321/', { waitUntil: 'networkidle0', timeout: 30000 });
await new Promise(r => setTimeout(r, 800));
// click Interactive DT group title to expand
const collapsedBefore = await page.evaluate(() => document.querySelectorAll('.kb-nav .node.expanded').length);
await page.click('.kb-nav .node[data-depth="0"][data-bold] .group-title');
await new Promise(r => setTimeout(r, 200));
const collapsedAfter = await page.evaluate(() => document.querySelectorAll('.kb-nav .node.expanded').length);
console.log('  expanded nodes:', collapsedBefore, '→', collapsedAfter);
// click AI button
console.log('→ AI button');
await page.click('#ai-open-btn');
await new Promise(r => setTimeout(r, 500));
const aiOpen = await page.evaluate(() => document.getElementById('ai-sidebar')?.classList.contains('open'));
console.log('  ai sidebar open:', aiOpen);
// click again to close
await page.click('#ai-open-btn');
await new Promise(r => setTimeout(r, 500));
// click Edit button
console.log('→ Edit button');
await page.click('#edit-btn');
await new Promise(r => setTimeout(r, 400));
const editOpen = await page.evaluate(() => document.getElementById('editor-panel')?.classList.contains('open'));
console.log('  editor panel open:', editOpen);
// navigate to a paper card
console.log('→ Paper card (ArtGS)');
await page.goto('http://localhost:4321/idea-research/ideas/interactive-dt/wiki/papers/artgs/', { waitUntil: 'networkidle0', timeout: 30000 });
await new Promise(r => setTimeout(r, 800));
const sidebarActive = await page.evaluate(() => document.querySelector('.kb-nav a.active')?.textContent);
console.log('  active sidebar item:', sidebarActive);
const tocItems = await page.evaluate(() => document.querySelectorAll('#kb-toc a').length);
console.log('  TOC items:', tocItems);
const pdfIframe = await page.evaluate(() => document.querySelector('iframe.pdf-embed')?.src || 'none');
console.log('  PDF iframe src:', pdfIframe.replace('http://localhost:4321', ''));
await page.screenshot({ path: resolve(__dirname, '..', 'screenshots', 'real', 'final-paper.png'), fullPage: false });
await browser.close();
console.log('\nerrors:', errors.length ? errors : '(none)');
