(function () {
  "use strict";

  var state = { settings: null };

  var $ = function (id) { return document.getElementById(id); };
  var statusPill = $("status-pill");
  var saveFeedback = $("save-feedback");
  var claudeCard = $("claude-cli-card");
  var openaiCard = $("openai-card");
  var profilesEl = $("openai-profiles");
  var defaultModelSelect = $("openai-default-model");
  var claudeDefaultSelect = $("claude-default-model");

  function setSaveFeedback(text, type) {
    saveFeedback.textContent = text || "";
    saveFeedback.classList.remove("ok", "error");
    if (type) saveFeedback.classList.add(type);
  }

  function setPill(text, type) {
    statusPill.textContent = text;
    statusPill.classList.remove("good", "bad");
    if (type) statusPill.classList.add(type);
  }

  async function loadSettings() {
    try {
      var resp = await fetch("/api/settings");
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      state.settings = await resp.json();
      renderAll();
      loadKbs();
    } catch (err) {
      setPill("加载失败: " + err.message, "bad");
    }
  }

  // ───── KB 管理 ─────
  var kbCardsEl = $("kb-cards");

  function renderKbs(items) {
    kbCardsEl.innerHTML = "";
    if (!items.length) {
      kbCardsEl.innerHTML = '<div class="kb-empty">还没有知识库，下面创建一个 →</div>';
      return;
    }
    items.forEach(function (kb) {
      var card = document.createElement("div");
      card.className = "kb-card";
      card.innerHTML =
        '<div class="kb-card-head">' +
          '<div>' +
            '<div class="kb-card-name"></div>' +
            '<div class="kb-card-meta"></div>' +
          '</div>' +
          '<div class="kb-card-actions">' +
            '<a class="btn-open" target="_blank">打开 ↗</a>' +
            '<button class="secondary kb-rename">改名</button>' +
            '<button class="danger kb-delete">删除</button>' +
          '</div>' +
        '</div>' +
        '<div class="kb-card-upload">' +
          '<label class="kb-upload-btn">' +
            '<span>＋ 上传文件</span>' +
            '<input type="file" multiple style="display:none">' +
          '</label>' +
          '<label class="kb-upload-btn secondary-btn">' +
            '<span>＋ 上传文件夹</span>' +
            '<input type="file" webkitdirectory multiple style="display:none">' +
          '</label>' +
          '<div class="kb-upload-status"></div>' +
        '</div>';
      card.querySelector(".kb-card-name").textContent = kb.name;
      card.querySelector(".kb-card-meta").textContent =
        kb.slug + " · " + kb.files + " 篇 · " + (kb.chars / 10000).toFixed(1) + " 万字";
      card.querySelector(".btn-open").href = "/#/kb/" + encodeURIComponent(kb.slug) + "/";

      card.querySelector(".kb-rename").addEventListener("click", function () {
        var nv = prompt("新的知识库名称：", kb.name);
        if (nv && nv.trim() && nv.trim() !== kb.name) renameKb(kb.slug, nv.trim());
      });
      card.querySelector(".kb-delete").addEventListener("click", function () {
        if (confirm("把「" + kb.name + "」整个目录搬到 _trash/？\n(_trash/ 在 .gitignore 内，本地保留，不会上推。)")) {
          deleteKb(kb.slug);
        }
      });

      var inputs = card.querySelectorAll("input[type=file]");
      var statusEl = card.querySelector(".kb-upload-status");
      inputs[0].addEventListener("change", function (e) {
        uploadFiles(kb.slug, e.target.files, false, statusEl);
        e.target.value = "";
      });
      inputs[1].addEventListener("change", function (e) {
        uploadFiles(kb.slug, e.target.files, true, statusEl);
        e.target.value = "";
      });

      kbCardsEl.appendChild(card);
    });
  }

  async function loadKbs() {
    try {
      var resp = await fetch("/api/kbs");
      var data = await resp.json();
      renderKbs(data.items || []);
    } catch (err) {
      kbCardsEl.innerHTML = '<div class="kb-empty error">加载失败：' + err.message + "</div>";
    }
  }

  async function renameKb(slug, newName) {
    try {
      var resp = await fetch("/api/kbs/" + encodeURIComponent(slug), {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: newName }),
      });
      var data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || "HTTP " + resp.status);
      loadKbs();
    } catch (err) {
      alert("改名失败：" + err.message);
    }
  }

  async function deleteKb(slug) {
    try {
      var resp = await fetch("/api/kbs/" + encodeURIComponent(slug), { method: "DELETE" });
      var data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || "HTTP " + resp.status);
      loadKbs();
    } catch (err) {
      alert("删除失败：" + err.message);
    }
  }

  async function uploadFiles(slug, fileList, fromFolder, statusEl) {
    var files = Array.from(fileList || []);
    if (!files.length) return;
    var form = new FormData();
    files.forEach(function (file) {
      form.append("files", file, file.name);
      form.append("relative_paths", fromFolder ? (file.webkitRelativePath || file.name) : file.name);
    });
    statusEl.textContent = "上传 " + files.length + " 个文件中…";
    statusEl.classList.remove("error");
    try {
      var resp = await fetch("/api/kbs/" + encodeURIComponent(slug) + "/upload", {
        method: "POST",
        body: form,
      });
      var data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || "HTTP " + resp.status);
      var msg = "已导入 " + (data.saved || []).length + " 个";
      if (data.errors && data.errors.length) {
        msg += "，失败 " + data.errors.length + " 个：" +
          data.errors.slice(0, 3).map(function (e) { return e.path + " (" + e.error + ")"; }).join("; ");
        statusEl.classList.add("error");
      }
      statusEl.textContent = msg;
      loadKbs();
    } catch (err) {
      statusEl.textContent = "上传失败：" + err.message;
      statusEl.classList.add("error");
    }
  }

  async function createKb() {
    var name = $("kb-new-name").value.trim();
    var slug = $("kb-new-slug").value.trim();
    var fb = $("kb-create-feedback");
    if (!name) {
      fb.textContent = "请输入名称";
      fb.classList.add("error");
      return;
    }
    fb.textContent = "创建中…";
    fb.classList.remove("error", "ok");
    try {
      var resp = await fetch("/api/kbs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name, slug: slug || null }),
      });
      var data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || "HTTP " + resp.status);
      $("kb-new-name").value = "";
      $("kb-new-slug").value = "";
      fb.textContent = "已创建：" + data.name;
      fb.classList.add("ok");
      loadKbs();
    } catch (err) {
      fb.textContent = "失败：" + err.message;
      fb.classList.add("error");
    }
  }

  function renderAll() {
    var s = state.settings;
    setPill(
      s.is_configured ? "当前 backend 已就绪" : "当前 backend 未就绪",
      s.is_configured ? "good" : "bad"
    );

    document.querySelectorAll('input[name="backend"]').forEach(function (radio) {
      radio.checked = radio.value === s.backend;
      radio.onchange = onBackendChange;
    });
    onBackendChange();

    // Claude CLI
    $("claude-cli-path").value = s.claude_cli.cli_path || "";
    $("claude-cli-detected").textContent = s.claude_cli_available
      ? "✓ 已在 PATH 上探测到 claude"
      : "✗ PATH 上未找到 claude，建议手动填路径或先 npm install -g @anthropic-ai/claude-code";
    claudeDefaultSelect.innerHTML = "";
    (s.claude_cli.models || []).forEach(function (m) {
      var opt = document.createElement("option");
      opt.value = m.key;
      opt.textContent = m.name + " (" + m.model + ")";
      claudeDefaultSelect.appendChild(opt);
    });
    claudeDefaultSelect.value = s.claude_cli.default_model_key || "";

    // OpenAI
    renderProfiles();
    $("openai-enable-tools").checked = !!s.openai_api.enable_tools;
    $("openai-allow-client-model").checked = !!s.openai_api.allow_client_model;

    // 对话默认
    var cd = s.chat_defaults || {};
    $("chat-effort").value = cd.effort != null ? cd.effort : "max";
    $("chat-enable-tools").checked = cd.enable_tools !== false;
    $("chat-with-page").checked = cd.with_page !== false;

    // Access password
    $("access-password").value = "";
    $("access-password-state").textContent = s.access_password_set
      ? "当前已设置密码（本机访问无需密码）"
      : "当前未设置密码";
  }

  function onBackendChange() {
    var backend = document.querySelector('input[name="backend"]:checked');
    var name = backend ? backend.value : "claude_cli";
    claudeCard.classList.toggle("dim", name !== "claude_cli");
    openaiCard.classList.toggle("dim", name !== "openai_api");
  }

  function renderProfiles() {
    var profiles = state.settings.openai_api.models || [];
    profilesEl.innerHTML = "";
    defaultModelSelect.innerHTML = "";
    profiles.forEach(function (p, idx) {
      var card = document.createElement("div");
      card.className = "profile-card";
      card.dataset.idx = idx;
      card.innerHTML =
        '<div class="header">' +
          '<h4>模型 #' + (idx + 1) + (p.configured ? ' · 已配置 key' : ' · 未配置') + '</h4>' +
          '<button type="button" class="danger small">删除</button>' +
        '</div>' +
        '<div class="row">' +
          '<div class="field"><label>显示名</label><input data-k="name" value=""></div>' +
          '<div class="field"><label>key (内部，唯一)</label><input data-k="key" value=""></div>' +
        '</div>' +
        '<div class="field"><label>实际模型 ID</label><input data-k="model"></div>' +
        '<div class="field"><label>API Base URL</label><input data-k="api_base_url"></div>' +
        '<div class="field"><label>API Key（已配置时显示 ****，留空或保留 **** = 不修改）</label><input type="password" data-k="api_key"></div>' +
        '<div class="field"><label>上下文窗口（用于前端显示）</label><input type="number" data-k="context"></div>';

      card.querySelector('[data-k="name"]').value = p.name || "";
      card.querySelector('[data-k="key"]').value = p.key || "";
      card.querySelector('[data-k="model"]').value = p.model || "";
      card.querySelector('[data-k="api_base_url"]').value = p.api_base_url || "";
      card.querySelector('[data-k="api_key"]').value = p.configured ? "****" : "";
      card.querySelector('[data-k="context"]').value = p.context || 200000;

      card.querySelector(".danger").addEventListener("click", function () {
        if (profiles.length <= 1) {
          alert("至少保留一个模型 profile，需要 API 后端的话。");
          return;
        }
        profiles.splice(idx, 1);
        renderProfiles();
      });

      profilesEl.appendChild(card);

      var opt = document.createElement("option");
      opt.value = p.key;
      opt.textContent = p.name + " (" + p.model + ")";
      defaultModelSelect.appendChild(opt);
    });
    defaultModelSelect.value = state.settings.openai_api.default_model_key || (profiles[0] && profiles[0].key) || "";
  }

  function collectProfiles() {
    return Array.from(profilesEl.querySelectorAll(".profile-card")).map(function (card) {
      var get = function (k) {
        var el = card.querySelector('[data-k="' + k + '"]');
        return el ? el.value : "";
      };
      var ctxRaw = parseInt(get("context"), 10);
      return {
        key: get("key").trim(),
        name: get("name").trim(),
        model: get("model").trim(),
        api_base_url: get("api_base_url").trim(),
        api_key: get("api_key"),  // **** 留给后端识别
        context: isNaN(ctxRaw) ? 200000 : ctxRaw,
      };
    });
  }

  function buildPayload() {
    var pwd = $("access-password").value;
    var oldPwdSet = state.settings.access_password_set;
    var passwordField;
    if (pwd === "REMOVE") {
      passwordField = "";
    } else if (pwd === "") {
      // 留空 = 保留原值。后端 PUT 永远写完整结构，因此我们必须给出旧值。
      // 由于后端 public_view 不返回真密码，我们只能传一个特殊 sentinel 让后端理解；
      // 这里用空串 + 让后端在 settings.py 里判断是否替换。简化方案：直接传空串
      // 会导致清空密码 —— 这是 bad UX。所以前端用一个固定 sentinel "__KEEP__"。
      passwordField = "__KEEP__";
    } else {
      passwordField = pwd;
    }
    return {
      backend: document.querySelector('input[name="backend"]:checked').value,
      access_password: passwordField,
      claude_cli: {
        cli_path: $("claude-cli-path").value.trim(),
        models: state.settings.claude_cli.models,
        default_model_key: claudeDefaultSelect.value,
        enable_tools: state.settings.claude_cli.enable_tools,
      },
      openai_api: {
        models: collectProfiles(),
        default_model_key: defaultModelSelect.value,
        allow_client_model: $("openai-allow-client-model").checked,
        enable_tools: $("openai-enable-tools").checked,
        include_usage: state.settings.openai_api.include_usage,
        request_timeout: state.settings.openai_api.request_timeout,
        max_tool_rounds: state.settings.openai_api.max_tool_rounds,
        temperature: state.settings.openai_api.temperature,
      },
      chat_defaults: {
        effort: $("chat-effort").value,
        enable_tools: $("chat-enable-tools").checked,
        with_page: $("chat-with-page").checked,
      },
    };
  }

  async function save() {
    setSaveFeedback("保存中…");
    try {
      var resp = await fetch("/api/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildPayload()),
      });
      var data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || "HTTP " + resp.status);
      state.settings = data;
      renderAll();
      setSaveFeedback("已保存 ✓", "ok");
    } catch (err) {
      setSaveFeedback("保存失败：" + err.message, "error");
    }
  }

  $("add-profile-btn").addEventListener("click", function () {
    state.settings.openai_api.models.push({
      key: "model-" + (state.settings.openai_api.models.length + 1),
      name: "新模型",
      model: "",
      api_base_url: "https://api.openai.com/v1",
      api_key: "",
      context: 200000,
      configured: false,
    });
    renderProfiles();
  });

  $("save-btn").addEventListener("click", save);
  var createBtn = $("kb-create-btn");
  if (createBtn) createBtn.addEventListener("click", createKb);

  loadSettings();
})();
