/**
 * FastPic 前端工具函数
 */

/** 转义 HTML 文本内容，防止 XSS */
function escapeHtml(s) {
    if (s == null) return '';
    var div = document.createElement('div');
    div.textContent = s;
    return div.innerHTML;
}

/** 转义 HTML 属性值，防止 XSS */
function escapeAttr(s) {
    if (s == null) return '';
    return String(s)
        .replace(/&/g, '&amp;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}

/** 格式化文件大小（字节 -> 可读字符串） */
function formatSize(bytes) {
    if (bytes === 0) return '0 B';
    var units = ['B', 'KB', 'MB', 'GB', 'TB'];
    var i = Math.floor(Math.log(bytes) / Math.log(1024));
    return (bytes / Math.pow(1024, i)).toFixed(1) + ' ' + units[i];
}

/**
 * 构建 gallery 请求 URL；opts 可覆盖 mode/sortBy/sortOrder/cols/filters（用于 marker 状态等）
 */
function buildGalleryUrl(path, opts) {
    opts = opts || {};
    var mode = opts.mode !== undefined ? opts.mode : ((document.getElementById('mode-input') && document.getElementById('mode-input').value) || 'folder');
    var sortBy = opts.sortBy !== undefined ? opts.sortBy : ((document.getElementById('sort-by-input') && document.getElementById('sort-by-input').value) || 'modified_at');
    var sortOrder = opts.sortOrder !== undefined ? opts.sortOrder : ((document.getElementById('sort-order-input') && document.getElementById('sort-order-input').value) || 'desc');
    var cols = opts.cols !== undefined ? opts.cols : ((document.getElementById('cols-input') && document.getElementById('cols-input').value) || '4');
    var filters = opts.filters !== undefined ? opts.filters : ((typeof window.getFilterState === 'function') ? window.getFilterState() : {});
    return '/gallery?path=' + encodeURIComponent(path || '') + '&search=&mode=' + mode + '&sort_by=' + sortBy + '&sort_order=' + sortOrder + '&page=1&cols=' + encodeURIComponent(cols) +
        '&filter_filename=' + encodeURIComponent(filters.filter_filename || '') +
        '&filter_size_min=' + encodeURIComponent(filters.filter_size_min || '') +
        '&filter_size_max=' + encodeURIComponent(filters.filter_size_max || '') +
        '&filter_date_from=' + encodeURIComponent(filters.filter_date_from || '') +
        '&filter_date_to=' + encodeURIComponent(filters.filter_date_to || '') +
        '&filter_tag=' + encodeURIComponent(filters.filter_tag || '');
}

/** 从 location.search 解析 path 参数 */
function getPathFromUrl() {
    var params = new URLSearchParams(location.search);
    return params.get('path') || '';
}
