/**
 * FastPic 通用文件夹浏览组件
 * 提供 breadcrumb + 子文件夹列表，统一使用 /api/subfolders
 */
function createFolderBrowser(breadcrumbEl, listEl, options) {
    options = options || {};
    var currentPath = options.initialPath || '';
    var itemClass = options.itemClass || 'folder-browse-item';
    var navClass = options.navClass || 'folder-browse-nav';
    var rootLabel = options.rootLabel || '全部文件夹';
    var emptyMsg = options.emptyMessage;
    var onPathChange = options.onPathChange || function() {};

    function getEmptyMessage() {
        if (emptyMsg) return emptyMsg;
        return currentPath ? '此文件夹下没有子文件夹' : '暂无文件夹';
    }

    function renderBreadcrumb() {
        var parts = currentPath ? currentPath.split('/') : [];
        var html = '<a href="#" class="' + navClass + '" data-path="">' + (typeof escapeHtml === 'function' ? escapeHtml(rootLabel) : rootLabel.replace(/</g, '&lt;')) + '</a>';
        for (var i = 0; i < parts.length; i++) {
            var path = parts.slice(0, i + 1).join('/');
            var esc = (typeof escapeAttr === 'function' ? escapeAttr(path) : path.replace(/"/g, '&quot;'));
            var partEsc = (typeof escapeHtml === 'function' ? escapeHtml(parts[i]) : parts[i].replace(/</g, '&lt;'));
            html += '<span class="text-slate-400"> › </span>';
            html += '<a href="#" class="' + navClass + '" data-path="' + esc + '">' + partEsc + '</a>';
        }
        breadcrumbEl.innerHTML = html;
        breadcrumbEl.querySelectorAll('.' + navClass).forEach(function(a) {
            a.addEventListener('click', function(e) {
                e.preventDefault();
                currentPath = a.getAttribute('data-path') || '';
                renderBreadcrumb();
                loadSubfolders();
                onPathChange(currentPath);
            });
        });
    }

    var api = {};
    function loadSubfolders() {
        var prepend = options.prependHtml;
        var prependHtml = (typeof prepend === 'function' ? prepend(api) : prepend) || '';
        listEl.innerHTML = prependHtml + '<div class="text-slate-400 text-sm py-4">加载中...</div>';
        fetch('/api/subfolders?path=' + encodeURIComponent(currentPath))
            .then(function(r) { return r.json(); })
            .then(function(data) {
                var subs = data.subfolders || [];
                var content = '';
                if (subs.length === 0) {
                    content = '<div class="text-slate-400 text-sm py-4">' + getEmptyMessage() + '</div>';
                } else {
                    content = subs.map(function(s) {
                        var pathEsc = (typeof escapeAttr === 'function' ? escapeAttr(s.full_path || '') : (s.full_path || '').replace(/"/g, '&quot;'));
                        var nameEsc = (typeof escapeHtml === 'function' ? escapeHtml(s.name || '') : (s.name || '').replace(/</g, '&lt;'));
                        return '<div class="' + itemClass + ' flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-slate-50 cursor-pointer transition-colors" data-path="' + pathEsc + '">' +
                            '<svg class="w-5 h-5 text-slate-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-width="2" stroke-linecap="round" stroke-linejoin="round" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"/></svg>' +
                            '<span class="font-medium text-slate-700">' + nameEsc + '</span>' +
                            (s.image_count > 0 ? '<span class="text-xs text-slate-400 ml-auto">' + s.image_count + ' 张</span>' : '') +
                            '</div>';
                    }).join('');
                }
                listEl.innerHTML = prependHtml + content;
                listEl.querySelectorAll('.' + itemClass).forEach(function(el) {
                    el.addEventListener('click', function() {
                        currentPath = el.getAttribute('data-path') || '';
                        renderBreadcrumb();
                        loadSubfolders();
                        onPathChange(currentPath);
                    });
                });
                if (options.afterRender) options.afterRender(api);
            })
            .catch(function() {
                listEl.innerHTML = prependHtml + '<div class="text-red-500 text-sm py-4">加载失败</div>';
                if (options.afterRender) options.afterRender(api);
            });
    }

    api.getPath = function() { return currentPath; };
    api.setPath = function(p) { currentPath = p || ''; };
    api.refresh = function() { renderBreadcrumb(); loadSubfolders(); };
    api.init = function() { renderBreadcrumb(); loadSubfolders(); };
    return api;
}
