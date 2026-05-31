/* eslint-disable */
// =============================================================================
// client.ts — 全局浏览器运行时
// 由 BaseLayout.astro 用 <script> 直接 import，会被 Astro 打成单个 bundle。
//
// 职责：
//   1. 暗色模式（prefers-color-scheme + localStorage + Ctrl+Shift+L）
//   2. 阅读进度条（顶部 2px）
//   3. 右侧 TOC（动态抓 .article h2/h3）+ scrollspy
//   4. 代码块 hover 复制按钮
//   5. Edit / AI 按钮 → 暂时跳 :8001 旧 Docsify（AI sidebar 真实接入是下一步）
// =============================================================================

const ROOT = document.documentElement;
const LS_MODE = 'kb_mode';

/* -------------------------------------------------------------------------- */
/* 1. dark mode                                                                */
/* -------------------------------------------------------------------------- */
function initMode() {
  let saved: string | null = null;
  try { saved = localStorage.getItem(LS_MODE); } catch {}
  if (saved !== 'light' && saved !== 'dark') saved = 'light';
  ROOT.dataset.mode = saved;
}

function setMode(mode: 'light' | 'dark') {
  ROOT.dataset.mode = mode;
  try { localStorage.setItem(LS_MODE, mode); } catch {}
  const btn = document.getElementById('mode-toggle');
  if (btn) btn.textContent = mode === 'dark' ? '☀' : '🌙';
}

function wireModeButton() {
  const btn = document.getElementById('mode-toggle');
  if (!btn) return;
  btn.textContent = ROOT.dataset.mode === 'dark' ? '☀' : '🌙';
  btn.addEventListener('click', () => {
    setMode(ROOT.dataset.mode === 'dark' ? 'light' : 'dark');
  });
}

/* -------------------------------------------------------------------------- */
/* 2. reading progress                                                         */
/* -------------------------------------------------------------------------- */
function wireProgressBar() {
  let bar = document.getElementById('kb-progress') as HTMLElement | null;
  if (!bar) {
    bar = document.createElement('div');
    bar.id = 'kb-progress';
    bar.className = 'progress-bar';
    document.body.appendChild(bar);
  }
  let ticking = false;
  function update() {
    const h = document.documentElement;
    const max = h.scrollHeight - h.clientHeight;
    bar!.style.width = (max > 0 ? (h.scrollTop / max) * 100 : 0) + '%';
    ticking = false;
  }
  window.addEventListener('scroll', () => {
    if (!ticking) { window.requestAnimationFrame(update); ticking = true; }
  }, { passive: true });
  update();
}

/* -------------------------------------------------------------------------- */
/* 3. right-side TOC                                                           */
/* -------------------------------------------------------------------------- */
function buildToc() {
  const nav = document.getElementById('kb-toc');
  if (!nav) return;
  nav.innerHTML = '';

  // 内容页：用 .article 的 h2/h3。首页（无 .article）：用 .main 的 h2.section-title
  // （专题 / 最近更新），让首页也呈现「左中右」三栏。
  const article = document.querySelector('.article');
  const scope = article || document.querySelector('.main');
  if (!scope) { nav.style.display = 'none'; return; }

  const headings = Array.from(
    scope.querySelectorAll(article ? 'h2, h3' : 'h2.section-title')
  ) as HTMLElement[];
  if (headings.length < 2) { nav.style.display = 'none'; return; }
  nav.style.display = '';

  const title = document.createElement('div');
  title.className = 'toc-title';
  title.textContent = '本页目录';
  nav.appendChild(title);

  if (!article) {
    const home = document.createElement('a');
    home.href = '#hero';
    home.textContent = '首页';
    nav.appendChild(home);
  }

  headings.forEach((h) => {
    if (!h.id) h.id = h.textContent!.trim().toLowerCase().replace(/[^\w一-鿿]+/g, '-');
    const a = document.createElement('a');
    a.href = '#' + h.id;
    a.textContent = h.textContent!.trim();
    if (h.tagName === 'H3') a.classList.add('h3');
    nav.appendChild(a);
  });

  updateTocActive();
}

function updateTocActive() {
  const nav = document.getElementById('kb-toc');
  if (!nav) return;
  // 内容页 scope=.article（h2/h3）；首页 scope=.main（h2.section-title）。
  const article = document.querySelector('.article');
  const scope = article || document.querySelector('.main');
  if (!scope) return;
  const headings = Array.from(
    scope.querySelectorAll(article ? 'h2, h3' : 'h2.section-title')
  ) as HTMLElement[];
  if (!headings.length) return;
  const scrollY = window.scrollY + 100;
  let activeIdx = 0;
  headings.forEach((h, i) => { if (h.offsetTop <= scrollY) activeIdx = i; });
  const links = nav.querySelectorAll('a');
  // 首页 buildToc 在 headings 前多插了一个「首页」锚（links[0]），headings[i] 对应
  // links[i+1]。内容页则一一对应（offset=0）。宁可少高亮（顶部停在「首页」）也别越界。
  const offset = article ? 0 : 1;
  links.forEach((a, i) => a.classList.toggle('active', i === activeIdx + offset));
}

let tocScrollTicking = false;
function wireTocScroll() {
  window.addEventListener('scroll', () => {
    if (!tocScrollTicking) {
      window.requestAnimationFrame(() => { updateTocActive(); tocScrollTicking = false; });
      tocScrollTicking = true;
    }
  }, { passive: true });
}

/* -------------------------------------------------------------------------- */
/* 4. code-block copy buttons                                                  */
/* -------------------------------------------------------------------------- */
function wireCopyButtons() {
  document.querySelectorAll('.article pre').forEach((pre) => {
    if (pre.querySelector('.kb-copy-btn')) return;
    const btn = document.createElement('button');
    btn.className = 'kb-copy-btn';
    btn.type = 'button';
    btn.textContent = '复制';
    btn.addEventListener('click', () => {
      const code = pre.querySelector('code');
      const text = code ? (code as HTMLElement).innerText : (pre as HTMLElement).innerText;
      navigator.clipboard?.writeText(text).then(
        () => { btn.textContent = '已复制'; setTimeout(() => (btn.textContent = '复制'), 1400); },
        () => { btn.textContent = '失败'; setTimeout(() => (btn.textContent = '复制'), 1400); }
      );
    });
    (pre as HTMLElement).style.position = 'relative';
    pre.appendChild(btn);
  });
}

/* -------------------------------------------------------------------------- */
/* 5. Keyboard shortcuts (mode toggle only; ai-sidebar.js / editor.js handle
 *    their own Ctrl+Shift+A / Ctrl+Shift+E bindings via #ai-open-btn / #edit-btn)
 * -------------------------------------------------------------------------- */
function wireActionButtons() {
  document.addEventListener('keydown', (e) => {
    const mod = e.ctrlKey || e.metaKey;
    if (mod && e.shiftKey && e.key.toLowerCase() === 'l') {
      e.preventDefault();
      setMode(ROOT.dataset.mode === 'dark' ? 'light' : 'dark');
    }
  });
}

/* -------------------------------------------------------------------------- */
/* boot                                                                         */
/* -------------------------------------------------------------------------- */
initMode(); // do this before paint to avoid flash

/* -------------------------------------------------------------------------- */
/* 6. Sidebar 折叠交互                                                          */
/* -------------------------------------------------------------------------- */
function wireSidebar() {
  // 点可折叠组的小标题 → 切换展开。默认展开状态由 SidebarItem 服务端渲染
  // （当前页所在分支展开），这里只处理用户主动点击。
  document.querySelectorAll('.sidebar .nav-subtitle').forEach((btn) => {
    btn.addEventListener('click', () => {
      const li = (btn as HTMLElement).closest('.nav-group-item') as HTMLElement | null;
      if (!li) return;
      const expanded = li.classList.toggle('expanded');
      btn.setAttribute('aria-expanded', expanded ? 'true' : 'false');
    });
  });
}

/* -------------------------------------------------------------------------- */
/* 7. PDF embed: 把 <iframe class="pdf-embed"> 包进 .pdf-embed-wrap + 拖动手柄。
 *    源自 Docsify 的 pdfEmbedPlugin，CSS 已在 global.css。
 * -------------------------------------------------------------------------- */
const LS_PDF_H = 'pdf_embed_h_v2';
function attachPdfResize(wrap: HTMLElement, iframe: HTMLIFrameElement) {
  // 宽度永远跟随 article（CSS .pdf-embed-wrap { width:100% }）——只持久化高度。
  // 旧版本（pdf_embed_size_v1）存过固定 width，会把 wrap 钉成固定 px、布局宽度一变
  // 就跟正文错位。换 key + 永不设 inline width，旧数据自动失效。
  wrap.style.removeProperty('width');
  try {
    const h = parseFloat(localStorage.getItem(LS_PDF_H) || '');
    if (h > 0) wrap.style.height = Math.max(320, h) + 'px';
  } catch {}
  function saveSize() {
    try {
      localStorage.setItem(LS_PDF_H, String(Math.round(wrap.getBoundingClientRect().height)));
    } catch {}
  }
  function makeHandle(cls: string, axis: 'x' | 'y' | 'xy') {
    const h = document.createElement('div');
    h.className = 'pdf-embed-resize ' + cls;
    h.addEventListener('mousedown', (ev: MouseEvent) => {
      ev.preventDefault();
      ev.stopPropagation();
      const rect = wrap.getBoundingClientRect();
      const startX = ev.clientX, startY = ev.clientY;
      const startW = rect.width, startH = rect.height;
      wrap.classList.add('resizing');
      document.body.style.userSelect = 'none';
      document.body.style.cursor = getComputedStyle(h).cursor;
      function onMove(e: MouseEvent) {
        if (axis !== 'y') wrap.style.width = Math.max(320, startW + (e.clientX - startX)) + 'px';
        if (axis !== 'x') wrap.style.height = Math.max(320, startH + (e.clientY - startY)) + 'px';
      }
      function onUp() {
        wrap.classList.remove('resizing');
        document.body.style.userSelect = '';
        document.body.style.cursor = '';
        document.removeEventListener('mousemove', onMove, true);
        document.removeEventListener('mouseup', onUp, true);
        saveSize();
        try { iframe.contentWindow?.dispatchEvent(new Event('resize')); } catch {}
      }
      document.addEventListener('mousemove', onMove, true);
      document.addEventListener('mouseup', onUp, true);
    });
    return h;
  }
  // 现在只保留底部高度调节（宽度跟 article 走，不需要横向调）
  wrap.appendChild(makeHandle('rz-s', 'y'));
}

function wirePdfEmbeds() {
  const article = document.querySelector('.article');
  if (!article) return;
  const iframe = article.querySelector('iframe.pdf-embed') as HTMLIFrameElement | null;
  if (!iframe) return;
  // 允许内嵌阅读器请求全屏（其工具栏「全屏」按钮调 requestFullscreen，需父页 iframe 授权）
  iframe.setAttribute('allow', 'fullscreen');
  iframe.setAttribute('allowfullscreen', '');
  // 已经被包过就跳过
  if (iframe.parentElement?.classList.contains('pdf-embed-wrap')) return;
  // marked 可能把 iframe 套在 <p> 里，先 unwrap
  const p = iframe.parentElement;
  if (p && p !== article && p.tagName === 'P') {
    article.insertBefore(iframe, p);
    if (p.children.length === 0 && !p.textContent?.trim()) p.remove();
  }
  // 包 wrap
  const wrap = document.createElement('div');
  wrap.className = 'pdf-embed-wrap';
  iframe.parentNode?.insertBefore(wrap, iframe);
  wrap.appendChild(iframe);
  attachPdfResize(wrap, iframe);
  // AI sidebar 用 PDF 全文做 context（如果 AI sidebar 已挂）
  try {
    const m = iframe.getAttribute('src')?.match(/[?&]pdf=([^&]+)/);
    if (m) {
      const pdfPath = decodeURIComponent(m[1]).replace(/\.pdf$/i, '');
      const ai = (window as any).__aiSidebar;
      if (ai?.setPagePath) ai.setPagePath(pdfPath);
    }
  } catch {}
}

/* -------------------------------------------------------------------------- */
/* 8. Edit 按钮智能隐藏：首页 / Astro-only 页（无 .article 真实 markdown 源）
 *    点 Edit 必然 404，所以直接藏掉。
 * -------------------------------------------------------------------------- */
function hideEditOnAstroOnlyPages() {
  const hasMd = !!document.querySelector('.article');
  if (!hasMd) {
    const b = document.getElementById('edit-btn');
    if (b) b.style.display = 'none';
  }
}

function boot() {
  wireModeButton();
  wireProgressBar();
  wireActionButtons();
  buildToc();
  wireTocScroll();
  wireCopyButtons();
  wireSidebar();
  wirePdfEmbeds();
  hideEditOnAstroOnlyPages();
}

if (document.readyState !== 'loading') boot();
else document.addEventListener('DOMContentLoaded', boot);

/* -------------------------------------------------------------------------- */
/* code-block copy-button CSS (injected here to keep client.ts self-contained) */
/* -------------------------------------------------------------------------- */
const style = document.createElement('style');
style.textContent = `
  .article pre .kb-copy-btn {
    position: absolute;
    top: 8px; right: 8px;
    border: 1px solid rgba(255, 255, 255, 0.16);
    background: rgba(255, 255, 255, 0.04);
    color: rgba(255, 255, 255, 0.7);
    font-family: var(--sans);
    font-size: 11px;
    font-weight: 500;
    padding: 3px 9px;
    border-radius: 5px;
    cursor: pointer;
    opacity: 0;
    transition: all 0.15s;
  }
  .article pre:hover .kb-copy-btn { opacity: 1; }
  .article pre .kb-copy-btn:hover {
    background: rgba(255, 255, 255, 0.10);
    color: #FFFFFF;
  }
`;
document.head.appendChild(style);
