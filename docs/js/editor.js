(function () {
  "use strict";

  var editBtn = document.getElementById("edit-btn");
  var panel = document.getElementById("editor-panel");
  var container = document.getElementById("editor-container");
  var filepathEl = document.getElementById("editor-filepath");
  var toolbar = document.querySelector(".editor-toolbar");
  var saveBtn = document.getElementById("editor-save");
  var closeBtn = document.getElementById("editor-close");
  var resizeHandle = document.getElementById("editor-resize");

  var editor = null;
  var currentFilePath = "";
  var originalContent = "";
  var isOpen = false;
  var hasSaved = false;

  function getPagePath() {
    var hash = window.location.hash || "";
    var path = hash.replace(/^#\/?/, "");
    path = path.split("?")[0];
    try { path = decodeURIComponent(path); } catch (_) {}
    return path || "";
  }

  // 找到当前可视区域对应的标题，在源码中定位
  function findVisibleHeading() {
    var headings = document.querySelectorAll(".markdown-section h1, .markdown-section h2, .markdown-section h3");
    var best = null;
    for (var i = 0; i < headings.length; i++) {
      var rect = headings[i].getBoundingClientRect();
      if (rect.top <= 100) {
        best = headings[i];
      } else {
        break;
      }
    }
    return best ? best.textContent.trim() : null;
  }

  function scrollEditorToHeading(headingText) {
    if (!editor || !headingText) return;
    // 在源码中搜索 # heading
    var lines = editor.getValue().split("\n");
    for (var i = 0; i < lines.length; i++) {
      var line = lines[i];
      // 匹配 # ## ### 等标题行
      var match = line.match(/^#{1,6}\s+(.+)/);
      if (match && match[1].trim() === headingText) {
        editor.scrollIntoView({ line: i, ch: 0 }, 100);
        editor.setCursor({ line: i, ch: 0 });
        return;
      }
    }
    // 如果没找到精确匹配，用页面滚动比例估算
    var scrollRatio = window.scrollY / (document.documentElement.scrollHeight - window.innerHeight || 1);
    var targetLine = Math.floor(scrollRatio * editor.lineCount());
    editor.scrollIntoView({ line: targetLine, ch: 0 }, 100);
    editor.setCursor({ line: targetLine, ch: 0 });
  }

  async function openEditor() {
    if (isOpen) return;
    var pagePath = getPagePath();
    var heading = findVisibleHeading();

    try {
      var resp = await fetch("/api/page-source?path=" + encodeURIComponent(pagePath));
      if (!resp.ok) throw new Error("Failed to load: " + resp.status);
      var data = await resp.json();

      currentFilePath = data.file_path;
      originalContent = data.source;
      hasSaved = false;
      filepathEl.textContent = currentFilePath;
      toolbar.classList.remove("modified");

      if (!editor) {
        editor = CodeMirror(container, {
          value: originalContent,
          mode: "gfm",
          theme: "one-dark",
          lineNumbers: true,
          lineWrapping: true,
          autoCloseBrackets: true,
          indentUnit: 2,
          tabSize: 2,
          indentWithTabs: false,
          extraKeys: {
            "Ctrl-S": function () { saveFile(); },
            "Cmd-S": function () { saveFile(); },
            "Escape": function () { closeEditor(); },
            "Tab": function (cm) {
              if (cm.somethingSelected()) {
                cm.indentSelection("add");
              } else {
                cm.replaceSelection("  ", "end");
              }
            },
          },
        });

        editor.on("change", function () {
          if (editor.getValue() !== originalContent) {
            toolbar.classList.add("modified");
          } else {
            toolbar.classList.remove("modified");
          }
        });
      } else {
        editor.setValue(originalContent);
        editor.clearHistory();
      }

      // 打开面板
      isOpen = true;
      panel.classList.add("open");
      document.body.classList.add("editor-open");

      setTimeout(function () {
        editor.refresh();
        editor.focus();
        scrollEditorToHeading(heading);
      }, 350);

    } catch (err) {
      alert("Error: " + err.message);
    }
  }

  async function saveFile() {
    if (!editor || !currentFilePath) return;

    var content = editor.getValue();
    if (content === originalContent) return;

    saveBtn.textContent = "Saving...";
    saveBtn.disabled = true;

    try {
      var resp = await fetch("/api/apply-edit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          file_path: currentFilePath,
          modified_content: content,
        }),
      });

      if (!resp.ok) throw new Error("Save failed: " + resp.status);

      originalContent = content;
      hasSaved = true;
      toolbar.classList.remove("modified");
      saveBtn.textContent = "Saved!";
      setTimeout(function () { saveBtn.textContent = "Save"; }, 1500);
    } catch (err) {
      alert("Save error: " + err.message);
      saveBtn.textContent = "Save";
    } finally {
      saveBtn.disabled = false;
    }
  }

  function closeEditor() {
    if (editor && editor.getValue() !== originalContent) {
      if (!confirm("Unsaved changes will be lost. Close?")) return;
    }

    isOpen = false;
    panel.classList.remove("open");
    document.body.classList.remove("editor-open");
    var content = document.querySelector(".content");
    if (content) content.style.marginRight = "";

    // 保存过则刷新 Docsify 页面
    if (hasSaved) {
      var hash = location.hash;
      location.hash = "";
      setTimeout(function () { location.hash = hash; }, 50);
    }
  }

  editBtn.addEventListener("click", function () {
    if (isOpen) closeEditor();
    else openEditor();
  });
  saveBtn.addEventListener("click", saveFile);
  closeBtn.addEventListener("click", closeEditor);

  // Ctrl+Shift+E toggle
  document.addEventListener("keydown", function (e) {
    if (e.ctrlKey && e.shiftKey && e.key === "E") {
      e.preventDefault();
      if (isOpen) closeEditor();
      else openEditor();
    }
  });

  // ========== Resize ==========
  if (resizeHandle) {
    var dragging = false;

    resizeHandle.addEventListener("mousedown", function (e) {
      e.preventDefault();
      dragging = true;
      resizeHandle.classList.add("dragging");
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
      panel.style.transition = "none";
    });

    document.addEventListener("mousemove", function (e) {
      if (!dragging) return;
      var newWidth = window.innerWidth - e.clientX;
      if (newWidth < 400) newWidth = 400;
      if (newWidth > window.innerWidth * 0.8) newWidth = window.innerWidth * 0.8;
      panel.style.width = newWidth + "px";
      var content = document.querySelector(".content");
      if (content) content.style.marginRight = newWidth + "px";
      if (editor) editor.refresh();
    });

    document.addEventListener("mouseup", function () {
      if (!dragging) return;
      dragging = false;
      resizeHandle.classList.remove("dragging");
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      panel.style.transition = "";
    });
  }
})();
