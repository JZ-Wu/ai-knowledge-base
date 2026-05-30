// Homepage stat-card injector. demo 同款：4 张卡，数字在上行（accent），单位在下行（muted）。
// 调 /api/stats（FastAPI on :8001，vite proxy 转发）。
// papers/ideas 字段后端目前没有，先用 fallback。
(function () {
  fetch((window.__KB_API_BASE || '') + '/api/stats').then(function (r) { return r.json(); }).then(function (d) {
    var el = document.getElementById('kb-stats');
    if (!el) return;
    var wan = (d.chars / 10000).toFixed(1);
    var papers = d.papers != null ? d.papers : 35;
    var ideas = d.ideas != null ? d.ideas : 12;
    el.innerHTML =
      '<div class="stat-card"><div class="stat-num">' + d.files + '</div><div class="stat-label">篇文档</div></div>' +
      '<div class="stat-card"><div class="stat-num">' + wan + ' 万</div><div class="stat-label">总字数</div></div>' +
      '<div class="stat-card"><div class="stat-num">' + papers + '</div><div class="stat-label">论文卡片</div></div>' +
      '<div class="stat-card"><div class="stat-num">' + ideas + '</div><div class="stat-label">活跃 idea</div></div>';
  }).catch(function () {});
})();
