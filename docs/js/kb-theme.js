/* =============================================================================
 * kb-theme.js — runtime helpers for Warm Material theme
 *
 * 加载时机：在 KB 的 index.html 中，必须在 <script src="docs/vendor/js/docsify.min.js">
 * 之前加载（这样我们 push 的 plugin 才能被 docsify 读到）。
 *
 * 提供的能力：
 *   1. 暗色模式切换（按系统偏好 + localStorage 持久化）
 *   2. 顶部阅读进度条
 *   3. 代码块复制按钮（hover 显示）
 *   4. 右侧 TOC（自动从 h2/h3 生成，scrollspy active）
 *
 * 启用条件：<html data-theme="warm-material">（其他主题下脚本仍会加载但不影响视觉）
 * ============================================================================= */
(function () {
  'use strict';

  var ROOT = document.documentElement;
  var LS_MODE = 'kb_theme_mode';

  /* ===========================================================================
   * 1. Dark mode toggle
   *    优先级：localStorage > prefers-color-scheme > 'light'
   * =========================================================================== */
  function initMode() {
    var saved = null;
    try { saved = localStorage.getItem(LS_MODE); } catch (_) {}
    if (!saved) {
      saved = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }
    ROOT.dataset.mode = saved;

    // listen to system changes — but only when user hasn't set a preference
    try {
      var mq = window.matchMedia('(prefers-color-scheme: dark)');
      if (mq.addEventListener) {
        mq.addEventListener('change', function (e) {
          var explicit = null;
          try { explicit = localStorage.getItem(LS_MODE); } catch (_) {}
          if (!explicit) ROOT.dataset.mode = e.matches ? 'dark' : 'light';
        });
      }
    } catch (_) {}
  }

  function setMode(mode) {
    ROOT.dataset.mode = mode;
    try { localStorage.setItem(LS_MODE, mode); } catch (_) {}
    var btn = document.getElementById('kb-mode-btn');
    if (btn) btn.textContent = mode === 'dark' ? '☀' : '🌙';
  }

  function injectModeButton() {
    if (document.getElementById('kb-mode-btn')) return;
    var holder = document.querySelector('.top-right-btns');
    // if KB doesn't have .top-right-btns container, create one
    if (!holder) {
      holder = document.createElement('div');
      holder.className = 'top-right-btns';
      document.body.appendChild(holder);
    }
    var btn = document.createElement('button');
    btn.id = 'kb-mode-btn';
    btn.className = 'kb-mode-btn';
    btn.title = '切换深色 / 浅色 (Ctrl+Shift+L)';
    btn.textContent = ROOT.dataset.mode === 'dark' ? '☀' : '🌙';
    btn.addEventListener('click', function () {
      setMode(ROOT.dataset.mode === 'dark' ? 'light' : 'dark');
    });
    // insert at the start so it sits before Edit/AI buttons
    holder.insertBefore(btn, holder.firstChild);
  }

  document.addEventListener('keydown', function (e) {
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key && e.key.toLowerCase() === 'l') {
      e.preventDefault();
      setMode(ROOT.dataset.mode === 'dark' ? 'light' : 'dark');
    }
  });

  /* ===========================================================================
   * 2. Reading progress bar
   * =========================================================================== */
  function injectProgressBar() {
    if (document.getElementById('kb-progress')) return;
    var el = document.createElement('div');
    el.id = 'kb-progress';
    el.className = 'kb-progress';
    document.body.appendChild(el);

    var ticking = false;
    function update() {
      var h = document.documentElement;
      var max = h.scrollHeight - h.clientHeight;
      var p = max > 0 ? (h.scrollTop / max) * 100 : 0;
      el.style.width = p + '%';
      ticking = false;
    }
    window.addEventListener('scroll', function () {
      if (!ticking) {
        window.requestAnimationFrame(update);
        ticking = true;
      }
    }, { passive: true });
    update();
  }

  /* ===========================================================================
   * 3. Code block copy buttons + lang label
   *    Docsify renders code as: <pre data-lang="python"><code class="lang-python">...
   *    We add a small button absolutely positioned inside <pre>.
   * =========================================================================== */
  function attachCopyButtons() {
    var pres = document.querySelectorAll('.markdown-section pre');
    pres.forEach(function (pre) {
      if (pre.querySelector('.kb-copy-btn')) return;
      var btn = document.createElement('button');
      btn.className = 'kb-copy-btn';
      btn.type = 'button';
      btn.textContent = '复制';
      btn.addEventListener('click', function () {
        var code = pre.querySelector('code');
        var text = code ? code.innerText : pre.innerText;
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text).then(function () {
            btn.textContent = '已复制';
            setTimeout(function () { btn.textContent = '复制'; }, 1400);
          }).catch(function () {
            btn.textContent = '失败';
            setTimeout(function () { btn.textContent = '复制'; }, 1400);
          });
        }
      });
      pre.appendChild(btn);
    });
  }

  /* ===========================================================================
   * 4. Right-side TOC (scrollspy)
   *    在大屏（>1280px）显示。从当前 .markdown-section 的 h2/h3 抓内容。
   * =========================================================================== */
  function buildToc() {
    // remove old TOC
    var old = document.getElementById('kb-toc');
    if (old) old.remove();

    if (window.innerWidth < 1280) return;

    var section = document.querySelector('.markdown-section');
    if (!section) return;
    var headings = section.querySelectorAll('h2, h3');
    if (headings.length < 2) return;  // not worth showing for tiny pages

    var nav = document.createElement('nav');
    nav.id = 'kb-toc';
    nav.className = 'kb-toc';

    var title = document.createElement('div');
    title.className = 'kb-toc-title';
    title.textContent = '本页目录';
    nav.appendChild(title);

    headings.forEach(function (h) {
      if (!h.id) return;
      var a = document.createElement('a');
      a.href = '#' + (location.hash.split('?')[0]) + '?id=' + h.id;
      a.textContent = h.textContent.replace(/^#+\s*/, '').trim();
      if (h.tagName === 'H3') a.classList.add('h3');
      a.dataset.target = h.id;
      a.addEventListener('click', function (e) {
        // let docsify handle navigation; we just smooth-scroll fallback
      });
      nav.appendChild(a);
    });

    document.body.appendChild(nav);
    updateTocActive();
  }

  function updateTocActive() {
    var nav = document.getElementById('kb-toc');
    if (!nav) return;
    var section = document.querySelector('.markdown-section');
    if (!section) return;
    var headings = Array.from(section.querySelectorAll('h2, h3')).filter(function (h) { return h.id; });
    if (!headings.length) return;
    var scrollY = window.scrollY + 100;
    var activeIdx = 0;
    headings.forEach(function (h, i) {
      if (h.offsetTop <= scrollY) activeIdx = i;
    });
    var links = nav.querySelectorAll('a');
    links.forEach(function (a, i) { a.classList.toggle('active', i === activeIdx); });
  }

  var tocScrollTicking = false;
  window.addEventListener('scroll', function () {
    if (!tocScrollTicking) {
      window.requestAnimationFrame(function () {
        updateTocActive();
        tocScrollTicking = false;
      });
      tocScrollTicking = true;
    }
  }, { passive: true });

  window.addEventListener('resize', function () {
    // rebuild TOC on resize across the 1280px breakpoint
    buildToc();
  });

  /* ===========================================================================
   * Docsify plugin — wire into the page-rendered lifecycle
   * =========================================================================== */
  function kbThemePlugin(hook) {
    hook.doneEach(function () {
      attachCopyButtons();
      buildToc();
    });
  }

  // Run pre-docsify init
  initMode();
  if (document.readyState !== 'loading') {
    injectModeButton();
    injectProgressBar();
  } else {
    document.addEventListener('DOMContentLoaded', function () {
      injectModeButton();
      injectProgressBar();
    });
  }

  // Register the docsify plugin (works whether $docsify is set yet or not)
  window.$docsify = window.$docsify || {};
  window.$docsify.plugins = (window.$docsify.plugins || []).concat(kbThemePlugin);
})();
