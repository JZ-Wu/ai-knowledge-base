(function () {
  "use strict";

  // ========== Models — single source of truth ==========
  // value : model id sent to backend (also used as MODEL_CONTEXT key in ai-sidebar.js)
  // label : shown in the <select> option
  // ctx   : context window in tokens, consumed by ai-sidebar.js's usage bar
  // Adding a new model? Add ONE entry here. ai-sidebar.js, index.html, and
  // pdf-reader.html all pick it up automatically.
  var MODELS = [
    { value: "claude-opus-4-8",           label: "Opus 4.8",   ctx: 1000000 },
    { value: "claude-opus-4-7",           label: "Opus 4.7",   ctx: 1000000 },
    { value: "claude-sonnet-4-6",         label: "Sonnet 4.6", ctx: 200000  },
    { value: "claude-haiku-4-5-20251001", label: "Haiku 4.5",  ctx: 200000  },
  ];
  var DEFAULT_MODEL = "claude-opus-4-8";

  window.AI_SIDEBAR_MODELS = MODELS;
  window.AI_SIDEBAR_DEFAULT_MODEL = DEFAULT_MODEL;

  // ========== Sidebar markup ==========

  function escapeAttr(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/"/g, "&quot;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function buildOptionsHtml(defaultModel) {
    return MODELS.map(function (m) {
      var sel = m.value === defaultModel ? " selected" : "";
      return '<option value="' + escapeAttr(m.value) + '"' + sel + '>' +
        escapeAttr(m.label) + "</option>";
    }).join("");
  }

  // ========== Models from /api/settings（取代硬编码，内置 MODELS 仅作离线兜底）==========
  function _ctxFor(p) {
    if (p.context) return p.context;
    return /opus/i.test(p.model || "") ? 1000000 : 200000;
  }
  function applyServerModels(settings) {
    if (!settings) return;
    // 思考强度默认值来自设置页 chat_defaults.effort（用户在 ai-sidebar.js 里改过则那边覆盖）
    var cd = settings.chat_defaults || {};
    if (cd.effort !== undefined) {
      var think = document.getElementById("ai-think");
      if (think) think.value = cd.effort;
    }
    var cfg = settings.backend === "openai_api" ? settings.openai_api : settings.claude_cli;
    var profiles = (cfg && cfg.models) || [];
    if (!profiles.length) return;
    MODELS = profiles.map(function (p) {
      return { value: p.model, label: p.name || p.model, ctx: _ctxFor(p) };
    });
    window.AI_SIDEBAR_MODELS = MODELS;
    var defProfile = profiles.filter(function (p) { return p.key === cfg.default_model_key; })[0] || profiles[0];
    if (defProfile) {
      DEFAULT_MODEL = defProfile.model;
      window.AI_SIDEBAR_DEFAULT_MODEL = DEFAULT_MODEL;
    }
    var sel = document.getElementById("ai-model");
    if (sel) {
      var prev = sel.value;
      sel.innerHTML = buildOptionsHtml(DEFAULT_MODEL);
      if (prev && MODELS.some(function (m) { return m.value === prev; })) sel.value = prev;
    }
  }
  function fetchServerModels() {
    var base = window.__KB_API_BASE || "";
    return fetch(base + "/api/settings", { credentials: "include" })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (s) { applyServerModels(s); })
      .catch(function () { /* 离线/未登录 → 用内置 MODELS 兜底 */ });
  }

  // mountAiSidebar — replace the slot element with the AI sidebar markup.
  //
  // opts:
  //   title             — header title (default "AI Assistant")
  //   placeholder       — textarea placeholder
  //   defaultModel      — pre-select this model in the dropdown
  //   initialQuotaText  — initial text inside #ai-quota-text (e.g. "Loading quota...")
  //   target            — DOM element or CSS selector (default "#ai-sidebar-slot")
  function mountAiSidebar(opts) {
    opts = opts || {};
    var title = opts.title || "AI Assistant";
    var placeholder = opts.placeholder || "Ask a question or request edits...";
    var defaultModel = opts.defaultModel || DEFAULT_MODEL;
    var initialQuotaText = opts.initialQuotaText || "";

    var html =
      '<div id="ai-sidebar" class="ai-sidebar">' +
        '<div id="ai-sidebar-resize" class="ai-sidebar-resize"></div>' +
        '<div class="ai-sidebar-header">' +
          '<span>' + escapeAttr(title) + '</span>' +
          '<div>' +
            '<button id="ai-clear" title="Clear history">&#x1F5D1;</button>' +
            '<button id="ai-close">&times;</button>' +
          '</div>' +
        '</div>' +
        '<div id="ai-quota-bar" style="padding:4px 16px;font-size:11px;color:#888;background:#f8f8f8;border-bottom:1px solid #e0e0e0;display:none;">' +
          '<span id="ai-quota-text">' + escapeAttr(initialQuotaText) + '</span>' +
        '</div>' +
        '<div id="ai-context-bar" style="padding:4px 16px;font-size:11px;color:#888;background:#f5f5f5;border-bottom:1px solid #e0e0e0;display:none;align-items:center;justify-content:space-between;">' +
          '<span id="ai-context-text"></span>' +
        '</div>' +
        '<div id="ai-context"></div>' +
        '<div id="ai-messages"></div>' +
        '<div class="ai-sidebar-input">' +
          '<div class="ai-model-row">' +
            '<select id="ai-model">' + buildOptionsHtml(defaultModel) + '</select>' +
            '<select id="ai-think" title="思考强度：Claude 走 --effort，OpenAI 推理模型走 reasoning_effort">' +
              '<option value="">思考：关闭</option>' +
              '<option value="low">思考：低</option>' +
              '<option value="medium" selected>思考：中</option>' +
              '<option value="high">思考：高</option>' +
              '<option value="max">思考：最高</option>' +
            '</select>' +
          '</div>' +
          '<div id="ai-usage-bar" style="font-size:11px;color:#999;margin-bottom:4px;display:none;"></div>' +
          '<div id="ai-quote-chip" class="ai-quote-chip" style="display:none"></div>' +
          '<textarea id="ai-input" placeholder="' + escapeAttr(placeholder) + '" rows="2"></textarea>' +
          '<div id="ai-image-preview" class="ai-image-preview"></div>' +
          '<div class="ai-btn-row">' +
            '<label class="ai-img-btn" title="Upload image">' +
              '&#128247;' +
              '<input type="file" id="ai-image-input" accept="image/*" style="display:none">' +
            '</label>' +
            '<button id="ai-send">Send</button>' +
          '</div>' +
        '</div>' +
      '</div>' +
      '<button id="ai-float-btn" class="ai-float-btn" style="display:none">Ask AI</button>' +
      '<button id="ai-quote-btn" class="ai-quote-btn" style="display:none">&#128172; 引用追问</button>';

    var target = opts.target || "#ai-sidebar-slot";
    var el = typeof target === "string" ? document.querySelector(target) : target;
    if (!el) {
      console.error("[ai-sidebar-mount] target not found:", target);
      return;
    }
    el.outerHTML = html;
  }

  window.mountAiSidebar = mountAiSidebar;

  // ========== Dependencies — single source of truth ==========
  //
  // ai-sidebar.js needs at runtime:
  //   - marked (saved as window.markedLib so docsify can keep its own window.marked)
  //   - DOMPurify (window.DOMPurify)
  //   - katex (window.katex, used by renderMarkdown for $...$ math)
  //
  // Previously these were loaded by each consumer page's own <script> chain,
  // so pdf-reader.html standalone forgot katex and math broke. Now the
  // sidebar declares its own deps here and bootstraps them itself.

  var DEPS = [
    {
      type: "css",
      href: "/docs/vendor/katex/katex.min.css",
      check: function () { return !!document.querySelector('link[href*="katex.min.css"]'); },
    },
    {
      type: "js",
      src: "/docs/vendor/katex/katex.min.js",
      check: function () { return typeof window.katex !== "undefined"; },
    },
    {
      type: "js",
      src: "/docs/vendor/js/marked.min.js",
      check: function () { return !!window.markedLib; },
      before: function () {
        // Save host page's window.marked (e.g. docsify's lexer) BEFORE
        // marked.min.js overrides it. Restored in after().
        if (window.marked && !window._docsifyMarked) {
          window._docsifyMarked = window.marked;
        }
      },
      after: function () {
        // Standalone marked v15 lives as markedLib; restore host's marked.
        if (window.marked && !window.markedLib) window.markedLib = window.marked;
        if (window._docsifyMarked) window.marked = window._docsifyMarked;
      },
    },
    {
      type: "js",
      src: "https://cdn.jsdelivr.net/npm/dompurify@3.2.4/dist/purify.min.js",
      check: function () { return typeof window.DOMPurify !== "undefined"; },
    },
    {
      type: "js",
      src: "/docs/js/ai-sidebar.js?v=13",
      check: function () { return !!window.__aiSidebarLoaded; },
    },
  ];

  function loadOne(dep) {
    return new Promise(function (resolve, reject) {
      if (dep.check && dep.check()) {
        if (dep.after) dep.after();
        return resolve();
      }
      if (dep.before) dep.before();
      var el;
      if (dep.type === "css") {
        el = document.createElement("link");
        el.rel = "stylesheet";
        el.href = dep.href;
      } else {
        el = document.createElement("script");
        el.src = dep.src;
      }
      el.onload = function () {
        if (dep.after) dep.after();
        resolve();
      };
      el.onerror = function () {
        reject(new Error("ai-sidebar dep failed: " + (dep.src || dep.href)));
      };
      (dep.type === "css" ? document.head : document.body).appendChild(el);
    });
  }

  // bootstrapAiSidebar — mount markup synchronously, then sequentially ensure
  // every runtime dep is present, then load ai-sidebar.js. Idempotent: deps
  // already loaded by the host page (e.g. docsify-katex) are skipped.
  function bootstrapAiSidebar(opts) {
    mountAiSidebar(opts);
    // 先用 /api/settings 的模型列表覆盖下拉（在加载 ai-sidebar.js 之前，
    // 这样它初始化时读到的 window.AI_SIDEBAR_MODELS 已是最新）。
    return fetchServerModels().then(function () {
      return DEPS.reduce(function (p, dep) {
        return p.then(function () { return loadOne(dep); });
      }, Promise.resolve());
    }).catch(function (err) {
      console.error("[ai-sidebar bootstrap]", err);
    });
  }

  window.bootstrapAiSidebar = bootstrapAiSidebar;
})();
