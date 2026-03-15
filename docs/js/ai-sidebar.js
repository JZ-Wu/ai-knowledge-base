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

  // ========== History Persistence ==========

  function saveHistory() {
    try {
      var data = { messages: chatMessages, page: currentPagePath };
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
    sessionStorage.removeItem(STORAGE_KEY);
  }

  // 页面加载时恢复历史
  loadHistory();

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
    currentPagePath = getPagePath();

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
    var content = document.querySelector(".content");
    if (content) content.style.marginRight = "";
  }

  closeBtn.addEventListener("click", closeSidebar);
  if (clearBtn) clearBtn.addEventListener("click", clearHistory);

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
    // 为 thinking 和正文创建独立容器，避免 renderMarkdown 覆盖 thinking
    var thinkContainer = document.createElement("div");
    thinkContainer.className = "ai-thinking-container";
    assistantEl.appendChild(thinkContainer);
    var textContainer = document.createElement("div");
    textContainer.className = "ai-text-container";
    assistantEl.appendChild(textContainer);

    var typingEl = showTyping(textContainer);

    abortController = new AbortController();
    setStreamingUI(true);

    try {
      var response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          page_path: currentPagePath,
          selected_text: selectedText,
          messages: chatMessages,
          model: modelSelect ? modelSelect.value : "claude-opus-4-6",
          thinking: thinkingCheckbox ? thinkingCheckbox.checked : false,
          images: images,
        }),
        signal: abortController.signal,
      });

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

            if (data.type === "thinking") {
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
              var toolInfo = document.createElement("div");
              toolInfo.className = "ai-tool-info";
              toolInfo.innerHTML =
                '<span class="tool-icon">&#9881;</span> ' +
                "<strong>" + escapeHtml(data.tool) + "</strong> " +
                '<span class="tool-file">' + escapeHtml(data.file || "") + "</span>";
              messagesEl.appendChild(toolInfo);
              scrollToBottom();
            } else if (data.type === "error") {
              fullResponse += "\n\n**Error:** " + data.content;
              renderMarkdown(textContainer, fullResponse);
            }
          } catch (_) {}
        }
      }

      // 回复完成后折叠 thinking
      var doneThinkEl = thinkContainer.querySelector(".ai-thinking-block");
      if (doneThinkEl) doneThinkEl.removeAttribute("open");

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

  function appendMessageDOM(role, text, images) {
    var el = document.createElement("div");
    el.className = "ai-msg " + role;
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
    messagesEl.appendChild(el);
    scrollToBottom();
    return el;
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
    if (typeof marked !== "undefined" && text) {
      // 保护 LaTeX 块不被 marked 破坏
      var mathBlocks = [];
      var placeholder = function (m) {
        var idx = mathBlocks.length;
        mathBlocks.push(m);
        return "\x00MATH" + idx + "\x00";
      };
      // 先提取 $$...$$ 和 \[...\]（display），再提取 $...$ 和 \(...\)（inline）
      // 注意：$$ 必须先于 $ 提取，避免误匹配
      var safe = text
        .replace(/\$\$([\s\S]+?)\$\$/g, placeholder)
        .replace(/\\\[([\s\S]+?)\\\]/g, placeholder)
        .replace(/\\\((.+?)\\\)/g, placeholder)
        .replace(/\$([^\$\n]+?)\$/g, placeholder);

      var html = marked.parse(safe);

      // 还原 LaTeX 块
      html = html.replace(/\x00MATH(\d+)\x00/g, function (_, idx) {
        return mathBlocks[parseInt(idx)];
      });
      el.innerHTML = html;
    } else {
      el.textContent = text;
    }
    renderKatex(el);
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
    });
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
