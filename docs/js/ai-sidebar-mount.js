(function () {
  "use strict";

  // ========== Models — single source of truth ==========
  // value : model id sent to backend (also used as MODEL_CONTEXT key in ai-sidebar.js)
  // label : shown in the <select> option
  // ctx   : context window in tokens, consumed by ai-sidebar.js's usage bar
  // Adding a new model? Add ONE entry here. ai-sidebar.js, index.html, and
  // pdf-reader.html all pick it up automatically.
  var MODELS = [
    { value: "claude-opus-4-7",           label: "Opus 4.7",   ctx: 1000000 },
    { value: "claude-opus-4-6",           label: "Opus 4.6",   ctx: 1000000 },
    { value: "claude-sonnet-4-6",         label: "Sonnet 4.6", ctx: 200000  },
    { value: "claude-haiku-4-5-20251001", label: "Haiku 4.5",  ctx: 200000  },
  ];
  var DEFAULT_MODEL = "claude-opus-4-7";

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
            '<label class="ai-thinking-toggle">' +
              '<input type="checkbox" id="ai-thinking" checked> Thinking' +
            '</label>' +
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
      src: "/docs/js/ai-sidebar.js?v=10",
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
    return DEPS.reduce(function (p, dep) {
      return p.then(function () { return loadOne(dep); });
    }, Promise.resolve()).catch(function (err) {
      console.error("[ai-sidebar bootstrap]", err);
    });
  }

  window.bootstrapAiSidebar = bootstrapAiSidebar;
})();
