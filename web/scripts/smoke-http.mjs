#!/usr/bin/env node
/**
 * HTTP smoke test against a running FastAPI server (:8001 prod serving web/dist).
 * No browser/puppeteer needed — pure fetch. Verifies pages + every /api endpoint +
 * PDF static + security blocks + 404. Discovers the first KB slug at runtime so it
 * is KB-agnostic. Exits non-zero on any failure (CI-friendly).
 *
 *   node web/scripts/smoke-http.mjs                 # default http://localhost:8001
 *   node web/scripts/smoke-http.mjs http://host:port
 */
const BASE = (process.argv[2] || 'http://localhost:8001').replace(/\/$/, '');
let pass = 0, fail = 0;

async function chk(desc, path, want = 200, contains = null) {
  try {
    const r = await fetch(BASE + path, { redirect: 'manual' });
    let ok = r.status === want;
    let extra = '';
    if (ok && contains) {
      const body = await r.text();
      ok = body.includes(contains);
      extra = ok ? '' : ` (missing "${contains}")`;
    }
    console.log(`  ${ok ? 'PASS' : 'FAIL'} [${r.status}${ok ? '' : ' want ' + want}] ${desc}${extra}`);
    ok ? pass++ : fail++;
    return r;
  } catch (e) {
    console.log(`  FAIL [ERR ${e.message}] ${desc}`);
    fail++;
    return null;
  }
}

console.log(`HTTP smoke test → ${BASE}`);

// API endpoints
await chk('api models', '/api/models', 200, 'models');
await chk('api rate-limits', '/api/rate-limits', 200, 'rate_limits');
await chk('api settings', '/api/settings', 200, 'backend');
const kbsResp = await chk('api kbs', '/api/kbs', 200, 'items');

// Discover first KB slug for page checks
let slug = null;
try {
  const kbs = await (await fetch(BASE + '/api/kbs')).json();
  slug = (kbs.items && kbs.items[0] && kbs.items[0].slug) || null;
} catch { /* ignore */ }

await chk('home page', '/', 200);
if (slug) {
  await chk(`KB landing /${slug}/`, `/kb/${slug}/`, 200);
  await chk('page-source (KB landing md)', `/api/page-source?path=kb/${encodeURIComponent(slug)}/README`, 200, 'source');
} else {
  console.log('  WARN  no KB slug discovered — skipping KB page checks');
}

// Tooling + frontend assets
await chk('pdf-reader tool', '/docs/tools/pdf-reader.html', 200);
await chk('ai-sidebar.js', '/docs/js/ai-sidebar.js', 200);

// Security: blocked paths + 404
await chk('block /server/.env', '/server/.env', 403);
await chk('block /knowledge_bases raw', '/knowledge_bases/x/idea.md', 403);
await chk('404 nonexistent KB', '/kb/__nonexistent_xyz__/', 404);

console.log(`\nRESULT: ${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
