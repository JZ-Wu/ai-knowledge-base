(function () {
  "use strict";

  var STORAGE_KEY = "ai_sidebar_history";

  // ========== State ==========
  var selectedText = "";
  var currentPagePath = "";
  var chatMessages = []; // {role, content}
  var isStreaming = false;
  var filesEdited = false;
  var abortController = null;
  var sessionId = ""; // Claude CLI 会话 ID，用于 resume 多轮对话
  var lastSentPagePath = ""; // 上次发送给 AI 的页面路径，避免重复发送相同页面内容

  // ========== DOM refs ==========
  var sidebar = document.getElementById("ai-sidebar");
  var closeBtn = document.getElementById("ai-close");
  var clearBtn = document.getElementById("ai-clear");
  var contextEl = document.getElementById("ai-context");
  var messagesEl = document.getElementById("ai-messages");
  var inputEl = document.getElementById("ai-input");
  var sendBtn = document.getElementById("ai-send");
  var floatBtn = document.getElementById("ai-float-btn");
  var modelSelect = document.getElementById("ai-model");
  var thinkingCheckbox = document.getElementById("ai-thinking");
  var imageInput = document.getElementById("ai-image-input");
  var imagePreview = document.getElementById("ai-image-preview");
  var pendingImages = []; // [{base64, media_type}]
  var quotaBar = document.getElementById("ai-quota-bar");
  var quotaText = document.getElementById("ai-quota-text");
  var usageBar = document.getElementById("ai-usage-bar");
  var contextBar = document.getElementById("ai-context-bar");
  var contextText = document.getElementById("ai-context-text");
  var compactBtn = document.getElementById("ai-compact-btn");
  var lastContextTokens = 0; // 最近一次 input_tokens（近似上下文大小）

  // ========== History Persistence ==========

  function saveHistory() {
    try {
      var data = { messages: chatMessages, page: currentPagePath, sessionId: sessionId };
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(data));
    } catch (_) {}
  }

  function loadHistory() {
    try {
      var raw = sessionStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      var data = JSON.parse(raw);
      if (!data.messages || !data.messages.length) return;

      chatMessages = data.messages;
      currentPagePath = data.page || "";
      sessionId = data.sessionId || "";

      // 重建 DOM
      chatMessages.forEach(function (m) {
        var el = appendMessageDOM(m.role, m.content);
        if (m.role === "assistant") renderMarkdown(el, m.content);
      });
    } catch (_) {}
  }

  function clearHistory() {
    chatMessages = [];
    messagesEl.innerHTML = "";
    contextEl.innerHTML = "";
    selectedText = "";
    sessionId = "";
    lastSentPagePath = "";
    lastContextTokens = 0;
    updateContextBar();
    sessionStorage.removeItem(STORAGE_KEY);
  }

  // ========== Model Persistence ==========

  var MODEL_STORAGE_KEY = "ai_sidebar_model";

  function saveModel() {
    if (modelSelect) {
      localStorage.setItem(MODEL_STORAGE_KEY, modelSelect.value);
    }
  }

  function loadModel() {
    var saved = localStorage.getItem(MODEL_STORAGE_KEY);
    if (saved && modelSelect) {
      modelSelect.value = saved;
    }
  }

  if (modelSelect) {
    modelSelect.addEventListener("change", saveModel);
  }

  // 页面加载时恢复历史和模型选择
  loadHistory();
  loadModel();

  // ========== Quota Display ==========

  function formatResetTime(ts) {
    if (!ts) return "";
    var d = new Date(ts * 1000);
    var now = new Date();
    var diffMs = d - now;
    if (diffMs <= 0) return "now";
    var diffMin = Math.floor(diffMs / 60000);
    if (diffMin < 60) return diffMin + "m";
    var diffH = Math.floor(diffMin / 60);
    var remainMin = diffMin % 60;
    if (diffH < 24) return diffH + "h" + (remainMin > 0 ? remainMin + "m" : "");
    var diffD = Math.floor(diffH / 24);
    return diffD + "d" + (diffH % 24) + "h";
  }

  function fetchQuota() {
    fetch("/api/rate-limits")
      .then(function(r) { return r.json(); })
      .then(function(data) {
        var rl = data.rate_limits;
        if (rl.error) {
          if (quotaBar) quotaBar.style.display = "none";
          return;
        }
        var parts = [];
        if (rl.five_hour) {
          var reset5 = formatResetTime(rl.five_hour.resets_at);
          parts.push("5h: " + rl.five_hour.used_percentage + "%" + (reset5 ? " (" + reset5 + ")" : ""));
        }
        if (rl.seven_day) {
          var reset7 = formatResetTime(rl.seven_day.resets_at);
          parts.push("7d: " + rl.seven_day.used_percentage + "%" + (reset7 ? " (" + reset7 + ")" : ""));
        }
        if (parts.length && quotaBar && quotaText) {
          quotaText.textContent = parts.join(" | ");
          quotaBar.style.display = "block";
          var maxPct = Math.max(
            rl.five_hour ? rl.five_hour.used_percentage : 0,
            rl.seven_day ? rl.seven_day.used_percentage : 0
          );
          quotaText.style.color = maxPct > 80 ? "#e53935" : maxPct > 50 ? "#f57c00" : "#888";
        }
      })
      .catch(function() {});
  }

  fetchQuota();

  // ========== Context Usage Display ==========

  // 模型 context window 大小（tokens）
  var MODEL_CONTEXT = {
    "claude-opus-4-6": 1000000,
    "claude-sonnet-4-6": 200000,
    "claude-haiku-4-5-20251001": 200000,
  };

  function updateContextBar() {
    if (!contextBar || !contextText) return;
    if (!lastContextTokens) {
      contextBar.style.display = "none";
      return;
    }
    var model = modelSelect ? modelSelect.value : "claude-sonnet-4-6";
    var maxCtx = MODEL_CONTEXT[model] || 200000;
    var pct = (lastContextTokens / maxCtx * 100).toFixed(1);
    var kTokens = (lastContextTokens / 1000).toFixed(1);
    var maxK = (maxCtx / 1000).toFixed(0);
    contextText.textContent = "Context: " + kTokens + "k / " + maxK + "k (" + pct + "%)";
    contextBar.style.display = "flex";
    var p = parseFloat(pct);
    contextText.style.color = p > 80 ? "#e53935" : p > 60 ? "#f57c00" : "#888";
    if (compactBtn) {
      compactBtn.disabled = isStreaming || !sessionId;
      compactBtn.style.display = sessionId ? "inline-block" : "none";
    }
  }

  if (compactBtn) {
    compactBtn.addEventListener("click", function() {
      if (!sessionId || isStreaming) return;
      doCompact();
    });
  }

  async function doCompact() {
    if (!sessionId) return;
    compactBtn.textContent = "Compacting...";
    compactBtn.disabled = true;

    try {
      var model = modelSelect ? modelSelect.value : "claude-sonnet-4-6";
      var response = await fetch("/api/compact", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, model: model }),
      });

      if (!response.ok) throw new Error("Compact failed: " + response.status);

      var reader = response.body.getReader();
      var decoder = new TextDecoder();
      var buffer = "";

      while (true) {
        var result = await reader.read();
        if (result.done) break;
        buffer += decoder.decode(result.value, { stream: true });
        var lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

        for (var i = 0; i < lines.length; i++) {
          var line = lines[i];
          if (!line.startsWith("data: ")) continue;
          try {
            var data = JSON.parse(line.slice(6));
            if (data.type === "session_id") {
              sessionId = data.session_id;
              saveHistory();
            } else if (data.type === "usage") {
              lastContextTokens = (data.input_tokens || 0) + (data.cache_read || 0) + (data.cache_create || 0);
              updateContextBar();
            } else if (data.type === "context_compact") {
              // 压缩成功
              var noteEl = document.createElement("div");
              noteEl.className = "ai-refresh-hint";
              noteEl.textContent = "Context compacted successfully.";
              messagesEl.appendChild(noteEl);
              scrollToBottom();
            }
          } catch (_) {}
        }
      }

      compactBtn.textContent = "Done!";
      setTimeout(function() { compactBtn.textContent = "Compact"; }, 1500);
    } catch (err) {
      compactBtn.textContent = "Failed";
      setTimeout(function() { compactBtn.textContent = "Compact"; }, 1500);
    } finally {
      compactBtn.disabled = false;
    }
  }

  // ========== A. Text Selection Detection ==========

  document.addEventListener("mouseup", function (e) {
    if (sidebar.contains(e.target) || floatBtn.contains(e.target)) return;

    setTimeout(function () {
      var sel = window.getSelection();
      var text = sel ? sel.toString().trim() : "";

      if (text.length > 0) {
        selectedText = text;
        var range = sel.getRangeAt(0);
        var rect = range.getBoundingClientRect();
        floatBtn.style.top = window.scrollY + rect.bottom + 6 + "px";
        floatBtn.style.left = window.scrollX + rect.right - 60 + "px";
        floatBtn.style.display = "block";
      } else {
        floatBtn.style.display = "none";
      }
    }, 10);
  });

  document.addEventListener("mousedown", function (e) {
    if (e.target === floatBtn || floatBtn.contains(e.target)) return;
    floatBtn.style.display = "none";
  });

  floatBtn.addEventListener("click", function (e) {
    e.preventDefault();
    e.stopPropagation();
    floatBtn.style.display = "none";
    openSidebar(selectedText);
  });

  // Fixed top-right open button
  var openBtn = document.getElementById("ai-open-btn");
  if (openBtn) {
    openBtn.addEventListener("click", function () {
      openSidebar("");
    });
  }

  // ========== B. Sidebar Control ==========

  function openSidebar(text) {
    sidebar.classList.add("open");
    document.body.classList.add("ai-sidebar-open");
    localStorage.setItem("ai-sidebar-open", "1");

    // 恢复保存的宽度
    var savedW = localStorage.getItem("ai-sidebar-width");
    if (savedW) {
      sidebar.style.width = savedW;
      var content = document.querySelector(".content");
      if (content) content.style.marginRight = savedW;
    }

    // 跟踪当前页面和选中文字（不重置 session，保留跨页对话上下文）
    var newPage = getPagePath();
    if (newPage !== currentPagePath) {
      currentPagePath = newPage;
    }
    if (text && text !== selectedText) {
      selectedText = text;
    }

    if (text) {
      contextEl.innerHTML =
        '<div class="context-label">Selected:</div>' +
        '<div class="context-text">' + escapeHtml(text) + "</div>";
    }
    inputEl.focus();
  }

  function closeSidebar() {
    sidebar.classList.remove("open");
    document.body.classList.remove("ai-sidebar-open");
    localStorage.setItem("ai-sidebar-open", "0");
    var content = document.querySelector(".content");
    if (content) content.style.marginRight = "";
  }

  closeBtn.addEventListener("click", closeSidebar);
  if (clearBtn) clearBtn.addEventListener("click", clearHistory);

  // Docsify SPA 页面切换时更新当前页面路径（保留 session 以支持跨页对话）
  window.addEventListener("hashchange", function () {
    var newPage = getPagePath();
    if (newPage !== currentPagePath) {
      currentPagePath = newPage;
    }
  });

  // Restore sidebar state from localStorage
  if (localStorage.getItem("ai-sidebar-open") === "1") {
    openSidebar("");
  }

  // Ctrl+Shift+A toggle
  document.addEventListener("keydown", function (e) {
    if (e.ctrlKey && e.shiftKey && e.key === "A") {
      e.preventDefault();
      if (sidebar.classList.contains("open")) closeSidebar();
      else openSidebar(selectedText);
    }
    if (e.key === "Escape" && sidebar.classList.contains("open")) {
      closeSidebar();
    }
  });

  // ========== Image Handling ==========

  function addImageFile(file) {
    if (!file || !file.type.startsWith("image/")) return;
    var reader = new FileReader();
    reader.onload = function (e) {
      var dataUrl = e.target.result;
      var base64 = dataUrl.split(",")[1];
      var media_type = file.type;
      pendingImages.push({ base64: base64, media_type: media_type });
      var thumb = document.createElement("div");
      thumb.className = "ai-img-thumb";
      thumb.innerHTML =
        '<img src="' + dataUrl + '">' +
        '<button class="ai-img-remove">&times;</button>';
      thumb.querySelector(".ai-img-remove").addEventListener("click", function () {
        var idx = Array.from(imagePreview.children).indexOf(thumb);
        if (idx >= 0) pendingImages.splice(idx, 1);
        thumb.remove();
      });
      imagePreview.appendChild(thumb);
    };
    reader.readAsDataURL(file);
  }

  if (imageInput) {
    imageInput.addEventListener("change", function () {
      if (this.files) {
        for (var i = 0; i < this.files.length; i++) addImageFile(this.files[i]);
      }
      this.value = "";
    });
  }

  // 粘贴图片
  inputEl.addEventListener("paste", function (e) {
    var items = e.clipboardData && e.clipboardData.items;
    if (!items) return;
    for (var i = 0; i < items.length; i++) {
      if (items[i].type.indexOf("image") !== -1) {
        addImageFile(items[i].getAsFile());
      }
    }
  });

  // ========== C. Chat Manager ==========

  sendBtn.addEventListener("click", function () {
    if (isStreaming) {
      cancelStreaming();
    } else {
      sendMessage();
    }
  });
  inputEl.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!isStreaming) sendMessage();
    }
  });

  function setStreamingUI(streaming) {
    isStreaming = streaming;
    if (streaming) {
      sendBtn.textContent = "Stop";
      sendBtn.classList.add("ai-stop-btn");
    } else {
      sendBtn.textContent = "Send";
      sendBtn.classList.remove("ai-stop-btn");
    }
  }

  function cancelStreaming() {
    if (abortController) {
      abortController.abort();
      abortController = null;
    }
  }

  async function sendMessage() {
    var text = inputEl.value.trim();
    if (!text && pendingImages.length === 0) return;
    if (isStreaming) return;

    var images = pendingImages.slice();
    pendingImages = [];
    imagePreview.innerHTML = "";

    inputEl.value = "";
    filesEdited = false;

    chatMessages.push({ role: "user", content: text });
    appendMessageDOM("user", text, images);
    saveHistory();

    var assistantEl = appendMessageDOM("assistant", "");
    // 为 thinking、工具调用和正文创建独立容器
    var thinkContainer = document.createElement("div");
    thinkContainer.className = "ai-thinking-container";
    assistantEl.appendChild(thinkContainer);
    var toolContainer = document.createElement("div");
    toolContainer.className = "ai-tool-container";
    assistantEl.appendChild(toolContainer);
    var textContainer = document.createElement("div");
    textContainer.className = "ai-text-container";
    assistantEl.appendChild(textContainer);

    var typingEl = showTyping(textContainer);

    abortController = new AbortController();
    setStreamingUI(true);

    try {
      // 有 session 时用 resume 模式，只发最后一条消息；否则发完整历史
      var sendMessages = sessionId
        ? [chatMessages[chatMessages.length - 1]]
        : chatMessages;
      var sendPagePath = currentPagePath;
      var response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          page_path: sendPagePath,
          selected_text: selectedText,
          messages: sendMessages,
          model: modelSelect ? modelSelect.value : "claude-opus-4-6",
          thinking: thinkingCheckbox ? thinkingCheckbox.checked : false,
          images: images,
          session_id: sessionId,
        }),
        signal: abortController.signal,
      });
      lastSentPagePath = currentPagePath;

      if (!response.ok) throw new Error("API error: " + response.status);

      var reader = response.body.getReader();
      var decoder = new TextDecoder();
      var buffer = "";
      var fullResponse = "";

      if (typingEl) { typingEl.remove(); typingEl = null; }

      while (true) {
        var result = await reader.read();
        if (result.done) break;

        buffer += decoder.decode(result.value, { stream: true });
        var lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

        for (var i = 0; i < lines.length; i++) {
          var line = lines[i];
          if (!line.startsWith("data: ")) continue;
          try {
            var data = JSON.parse(line.slice(6));

            if (data.type === "session_id") {
              sessionId = data.session_id;
              saveHistory();
            } else if (data.type === "thinking") {
              var thinkEl = thinkContainer.querySelector(".ai-thinking-block");
              if (!thinkEl) {
                thinkEl = document.createElement("details");
                thinkEl.className = "ai-thinking-block";
                thinkEl.setAttribute("open", "");
                thinkEl.innerHTML = "<summary>Thinking</summary><pre></pre>";
                thinkContainer.appendChild(thinkEl);
              }
              thinkEl.querySelector("pre").textContent = data.content;
              scrollToBottom();
            } else if (data.type === "text") {
              fullResponse += data.content;
              renderMarkdown(textContainer, fullResponse);
              scrollToBottom();
            } else if (data.type === "tool") {
              filesEdited = true;
              // 确保 toolContainer 里有 <details> 折叠块
              var toolDetails = toolContainer.querySelector(".ai-tool-block");
              if (!toolDetails) {
                toolDetails = document.createElement("details");
                toolDetails.className = "ai-tool-block";
                toolDetails.setAttribute("open", "");
                toolDetails.innerHTML = "<summary>Tool Calls</summary><div class='ai-tool-list'></div>";
                toolContainer.appendChild(toolDetails);
              }
              var toolList = toolDetails.querySelector(".ai-tool-list");
              var toolInfo = document.createElement("div");
              toolInfo.className = "ai-tool-info";
              toolInfo.innerHTML =
                '<span class="tool-icon">&#9881;</span> ' +
                "<strong>" + escapeHtml(data.tool) + "</strong> " +
                '<span class="tool-file">' + escapeHtml(data.file || "") + "</span>";
              toolList.appendChild(toolInfo);
              scrollToBottom();
            } else if (data.type === "context_compact") {
              fullResponse += "\n\n> *Context compacted — 上下文已压缩，早期对话可能被摘要*\n\n";
              renderMarkdown(textContainer, fullResponse);
              scrollToBottom();
            } else if (data.type === "error") {
              // resume 失败时重置 session，下次自动用完整历史新建会话
              if (sessionId && data.content.indexOf("CLI error") !== -1) {
                sessionId = "";
                saveHistory();
                fullResponse += "\n\n> Session 已过期，请重新发送消息。";
              } else {
                fullResponse += "\n\n**Error:** " + data.content;
              }
              renderMarkdown(textContainer, fullResponse);
            } else if (data.type === "usage") {
              // 上下文 ≈ input_tokens + cache_read + cache_create（总发送量）
              lastContextTokens = (data.input_tokens || 0) + (data.cache_read || 0) + (data.cache_create || 0);
              updateContextBar();
              if (usageBar) {
                var total = (data.input_tokens || 0) + (data.output_tokens || 0);
                var parts = [];
                parts.push("Tokens: " + total.toLocaleString());
                parts.push("In: " + (data.input_tokens || 0).toLocaleString());
                parts.push("Out: " + (data.output_tokens || 0).toLocaleString());
                if (data.cache_read) parts.push("Cache: " + data.cache_read.toLocaleString());
                usageBar.textContent = parts.join(" | ");
                usageBar.style.display = "block";
              }
            } else if (data.type === "duration") {
              if (usageBar) {
                usageBar.textContent += " | " + (data.ms / 1000).toFixed(1) + "s";
              }
            } else if (data.type === "rate_limit") {
              if (quotaBar && quotaText) {
                var resetStr = data.resets_at ? formatResetTime(data.resets_at) : "";
                var info = data.status || "";
                if (resetStr) info += (info ? " | " : "") + "Reset: " + resetStr;
                if (info) {
                  quotaText.textContent = info;
                  quotaBar.style.display = "block";
                  quotaText.style.color = data.status === "rate_limited" ? "#e53935" : "#888";
                }
              }
            }
          } catch (_) {}
        }
      }

      // 回复完成后折叠 thinking 和 tool calls
      var doneThinkEl = thinkContainer.querySelector(".ai-thinking-block");
      if (doneThinkEl) doneThinkEl.removeAttribute("open");
      var doneToolEl = toolContainer.querySelector(".ai-tool-block");
      if (doneToolEl) doneToolEl.removeAttribute("open");

      fetchQuota();

      chatMessages.push({ role: "assistant", content: fullResponse });
      saveHistory();

      if (filesEdited) {
        var refreshEl = document.createElement("div");
        refreshEl.className = "ai-refresh-hint";
        refreshEl.innerHTML =
          'Files modified. <a href="javascript:location.reload()">Refresh</a> to see changes.';
        messagesEl.appendChild(refreshEl);
        scrollToBottom();
      }
    } catch (err) {
      if (typingEl) typingEl.remove();
      if (err.name === "AbortError") {
        renderMarkdown(textContainer, fullResponse || "*[Cancelled]*");
      } else {
        renderMarkdown(textContainer, "**Error:** " + err.message);
      }
      if (fullResponse) {
        chatMessages.push({ role: "assistant", content: fullResponse });
        saveHistory();
      }
    } finally {
      abortController = null;
      setStreamingUI(false);
    }
  }

  function createCopyBtn(getText, align) {
    // align: "left" for user messages, "right" for assistant messages
    var btn = document.createElement("button");
    btn.className = "ai-msg-copy";
    btn.innerHTML = "&#128203;";
    btn.title = "Copy";
    var baseCss = "background:none;border:none;cursor:pointer;font-size:11px;padding:1px 4px;opacity:0;transition:opacity 0.15s;color:#999;";
    btn.style.cssText = baseCss;
    btn.addEventListener("click", function(e) {
      e.stopPropagation();
      var content = getText();
      if (!content) return;
      navigator.clipboard.writeText(content).then(function() {
        btn.innerHTML = "&#10003;";
        btn.style.color = "#4caf50";
        btn.style.opacity = "1";
        setTimeout(function() { btn.innerHTML = "&#128203;"; btn.style.color = "#999"; }, 1200);
      });
    });
    return btn;
  }

  function appendMessageDOM(role, text, images) {
    var el = document.createElement("div");
    el.className = "ai-msg " + role;
    el.style.position = "relative";

    if (role === "user") {
      if (images && images.length) {
        var imgRow = document.createElement("div");
        imgRow.style.cssText = "display:flex;gap:4px;margin-bottom:6px;flex-wrap:wrap;";
        images.forEach(function (img) {
          var imgEl = document.createElement("img");
          imgEl.src = "data:" + img.media_type + ";base64," + img.base64;
          imgEl.style.cssText = "max-height:80px;max-width:150px;border-radius:4px;";
          imgRow.appendChild(imgEl);
        });
        el.appendChild(imgRow);
      }
      if (text) {
        var textEl = document.createElement("span");
        textEl.textContent = text;
        el.appendChild(textEl);
      }
    }

    // 外部包装：消息气泡 + 气泡下方的操作栏
    var wrapper = document.createElement("div");
    wrapper.className = "ai-msg-wrapper " + role;

    // 操作栏在气泡外面下方
    var actionBar = document.createElement("div");
    actionBar.className = "ai-msg-actions";
    var align = role === "user" ? "flex-end" : "flex-start";
    actionBar.style.cssText = "display:flex;justify-content:" + align + ";gap:4px;padding:2px 4px 0;opacity:0;transition:opacity 0.15s;min-height:18px;";

    if (role === "user") {
      var copyBtn = createCopyBtn(function() { return text; });
      copyBtn.style.opacity = "1";
      copyBtn.style.color = "#aaa";
      actionBar.appendChild(copyBtn);

      var editBtn = document.createElement("button");
      editBtn.className = "ai-msg-edit";
      editBtn.innerHTML = "&#9998;";
      editBtn.title = "Edit";
      editBtn.style.cssText = "background:none;border:none;color:#aaa;cursor:pointer;font-size:11px;padding:1px 4px;";
      editBtn.addEventListener("click", function() { startEditMessage(el, text); });
      actionBar.appendChild(editBtn);
    } else {
      var copyBtn = createCopyBtn(function() {
        var assistantEls = messagesEl.querySelectorAll(".ai-msg.assistant");
        var elIdx = Array.from(assistantEls).indexOf(el);
        var count = 0;
        for (var j = 0; j < chatMessages.length; j++) {
          if (chatMessages[j].role === "assistant") {
            if (count === elIdx) return chatMessages[j].content;
            count++;
          }
        }
        return el.textContent || "";
      });
      copyBtn.style.opacity = "1";
      actionBar.appendChild(copyBtn);
    }

    wrapper.appendChild(el);
    wrapper.appendChild(actionBar);
    wrapper.addEventListener("mouseenter", function() { actionBar.style.opacity = "1"; });
    wrapper.addEventListener("mouseleave", function() { actionBar.style.opacity = "0"; });

    messagesEl.appendChild(wrapper);
    scrollToBottom();
    return el;
  }

  function startEditMessage(msgEl, originalText) {
    if (isStreaming) return;
    var msgIndex = -1;
    // Find the index of this message in chatMessages
    var userMsgs = messagesEl.querySelectorAll(".ai-msg.user");
    var userIdx = Array.from(userMsgs).indexOf(msgEl);
    if (userIdx < 0) return;
    // Map user message DOM index to chatMessages index
    var count = 0;
    for (var i = 0; i < chatMessages.length; i++) {
      if (chatMessages[i].role === "user") {
        if (count === userIdx) { msgIndex = i; break; }
        count++;
      }
    }
    if (msgIndex < 0) return;

    // Replace content with textarea
    var textarea = document.createElement("textarea");
    textarea.value = originalText;
    textarea.style.cssText = "width:100%;min-height:60px;border:1px solid rgba(255,255,255,0.3);border-radius:6px;padding:6px;font-size:14px;font-family:inherit;background:rgba(255,255,255,0.15);color:#fff;resize:vertical;box-sizing:border-box;";

    // Clear message content but keep structure
    var children = Array.from(msgEl.children);
    children.forEach(function(c) { c.style.display = "none"; });
    msgEl.appendChild(textarea);
    textarea.focus();

    var hint = document.createElement("div");
    hint.style.cssText = "font-size:11px;color:rgba(255,255,255,0.5);margin-top:4px;";
    hint.textContent = "Enter to submit, Escape to cancel";
    msgEl.appendChild(hint);

    textarea.addEventListener("keydown", function(e) {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        var newText = textarea.value.trim();
        if (!newText) return;
        // Remove this message and all subsequent messages
        chatMessages = chatMessages.slice(0, msgIndex);
        // Remove DOM elements from this message onward
        var allMsgs = messagesEl.querySelectorAll(".ai-msg");
        var startRemove = false;
        Array.from(allMsgs).forEach(function(m) {
          if (m === msgEl) startRemove = true;
          if (startRemove) m.remove();
        });
        // Also remove any refresh hints
        var hints = messagesEl.querySelectorAll(".ai-refresh-hint");
        hints.forEach(function(h) { h.remove(); });
        // Reset session since we're replaying
        sessionId = "";
        lastSentPagePath = "";
        // Re-send with edited text
        inputEl.value = newText;
        saveHistory();
        sendMessage();
      } else if (e.key === "Escape") {
        // Cancel edit
        textarea.remove();
        hint.remove();
        children.forEach(function(c) { c.style.display = ""; });
      }
    });
  }

  function showTyping(container) {
    var el = document.createElement("div");
    el.className = "ai-typing";
    el.innerHTML = "<span></span><span></span><span></span>";
    container.appendChild(el);
    scrollToBottom();
    return el;
  }

  function renderMarkdown(el, text) {
    var _marked = window.markedLib || window.marked;
    if (typeof _marked !== "undefined" && text) {
      // 提取 LaTeX 块，直接渲染为 HTML，避免 marked 和 renderMathInElement 二次解析冲突
      var mathHtmls = [];
      var placeholder = function (raw, display) {
        var idx = mathHtmls.length;
        var rendered;
        try {
          // 去掉定界符，提取纯 LaTeX 内容
          var tex = raw;
          if (display) {
            if (tex.startsWith("$$")) tex = tex.slice(2, -2);
            else if (tex.startsWith("\\[")) tex = tex.slice(2, -2);
          } else {
            if (tex.startsWith("\\(")) tex = tex.slice(2, -2);
            else if (tex.startsWith("$")) tex = tex.slice(1, -1);
          }
          rendered = katex.renderToString(tex.trim(), {
            displayMode: display,
            throwOnError: false,
            trust: true,
          });
        } catch (_) {
          rendered = '<code class="katex-error">' + escapeHtml(raw) + "</code>";
        }
        mathHtmls.push(rendered);
        return "\x00MATH" + idx + "\x00";
      };

      // 提取顺序很重要：先长定界符，后短定界符
      // 1) $$...$$ (display, 可跨行)
      // 2) \[...\] (display, 可跨行)
      // 3) \(...\) (inline)
      // 4) $...$ (inline, 不跨行，内容不能以空格开头/结尾，不匹配纯数字如价格 $100)
      var safe = text
        .replace(/\$\$([\s\S]+?)\$\$/g, function (m) { return placeholder(m, true); })
        .replace(/\\\[([\s\S]+?)\\\]/g, function (m) { return placeholder(m, true); })
        .replace(/\\\((.+?)\\\)/g, function (m) { return placeholder(m, false); })
        .replace(/\$([^\$\n]*?[^\$\s])\$/g, function (m, inner) {
          // 跳过空内容和纯数字 (如 $100)
          if (!inner || /^\d+$/.test(inner)) return m;
          return placeholder(m, false);
        });

      var html = _marked.parse(safe);

      // 还原已渲染的 KaTeX HTML
      html = html.replace(/\x00MATH(\d+)\x00/g, function (_, idx) {
        return mathHtmls[parseInt(idx)];
      });
      if (typeof DOMPurify !== 'undefined') {
        el.innerHTML = DOMPurify.sanitize(html, { ADD_TAGS: ['math', 'semantics', 'mrow', 'mi', 'mo', 'mn', 'msup', 'msub', 'mfrac', 'munder', 'mover', 'msqrt', 'mtext', 'annotation'], ADD_ATTR: ['encoding', 'xmlns'] });
      } else {
        el.innerHTML = html;
      }
    } else {
      el.textContent = text;
    }
    // 不再调用 renderMathInElement —— LaTeX 已在上面直接渲染完毕
  }

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  // ========== KaTeX ==========

  var katexDelimiters = [
    { left: "$$", right: "$$", display: true },
    { left: "\\[", right: "\\]", display: true },
    { left: "\\(", right: "\\)", display: false },
    { left: "$", right: "$", display: false },
  ];

  function renderKatex(el) {
    if (typeof renderMathInElement === "function") {
      try {
        renderMathInElement(el, { delimiters: katexDelimiters });
      } catch (_) {}
    }
  }

  // ========== Resize ==========

  var resizeHandle = document.getElementById("ai-sidebar-resize");
  if (resizeHandle) {
    var dragging = false;

    resizeHandle.addEventListener("mousedown", function (e) {
      e.preventDefault();
      dragging = true;
      resizeHandle.classList.add("dragging");
      document.body.classList.add("ai-sidebar-resizing");
    });

    document.addEventListener("mousemove", function (e) {
      if (!dragging) return;
      var newWidth = window.innerWidth - e.clientX;
      if (newWidth < 300) newWidth = 300;
      if (newWidth > window.innerWidth * 0.8) newWidth = window.innerWidth * 0.8;
      sidebar.style.width = newWidth + "px";
      document.querySelector(".content").style.marginRight = newWidth + "px";
    });

    document.addEventListener("mouseup", function () {
      if (!dragging) return;
      dragging = false;
      resizeHandle.classList.remove("dragging");
      document.body.classList.remove("ai-sidebar-resizing");
      localStorage.setItem("ai-sidebar-width", sidebar.style.width);
    });

    // 恢复保存的宽度
    var savedWidth = localStorage.getItem("ai-sidebar-width");
    if (savedWidth) {
      sidebar.style.width = savedWidth;
    }
  }

  // ========== Helpers ==========

  function getPagePath() {
    var hash = window.location.hash || "";
    var path = hash.replace(/^#\/?/, "");
    path = path.split("?")[0];
    try { path = decodeURIComponent(path); } catch (_) {}
    return path || "";
  }

  function escapeHtml(text) {
    var div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }
})();
