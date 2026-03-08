/**
 * FastPic 前端工具函数
 */

/** 防抖函数 */
function debounce(func, wait) {
    var timeout;
    return function executedFunction() {
        var context = this;
        var args = arguments;
        var later = function() {
            timeout = null;
            func.apply(context, args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

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
    var page = opts.page !== undefined ? opts.page : 1;
    var normalizedPath = path || '';
    var url = '/gallery?path=' + encodeURIComponent(normalizedPath) + '&search=&mode=' + mode + '&sort_by=' + sortBy + '&sort_order=' + sortOrder + '&page=' + page + '&cols=' + encodeURIComponent(cols) +
        '&filter_filename=' + encodeURIComponent(filters.filter_filename || '') +
        '&filter_size_min=' + encodeURIComponent(filters.filter_size_min || '') +
        '&filter_size_max=' + encodeURIComponent(filters.filter_size_max || '') +
        '&filter_date_from=' + encodeURIComponent(filters.filter_date_from || '') +
        '&filter_date_to=' + encodeURIComponent(filters.filter_date_to || '') +
        '&filter_tag=' + encodeURIComponent(filters.filter_tag || '');
    if (normalizedPath === '' && page === 1) {
        url += '&defer_subfolders=1';
    }
    return url;
}

/** 从 location.search 解析 path 参数 */
function getPathFromUrl() {
    var params = new URLSearchParams(location.search);
    return params.get('path') || '';
}

/**
 * 动态计算首屏显示的图片数量阈值
 * 基于视口高度和列数计算，适配不同设备和屏幕尺寸
 * 返回：首屏可显示的行数 * 列数
 */
function getFirstScreenThreshold() {
    var cols = 4;  // 默认列数
    var colsInput = document.getElementById('cols-input');
    if (colsInput) {
        cols = parseInt(colsInput.value, 10) || 4;
    }
    // 获取视口高度
    var viewportHeight = window.innerHeight || 600;
    // 估算每行高度（缩略图 + 间距 + 标题区域）
    // 文件夹模式：大约 200px 行高 + 16px gap
    // 假设每行约 220px
    var estimatedRowHeight = 220;
    // 减去顶部工具栏和底部空间（约 200px）
    var usableHeight = viewportHeight - 200;
    // 计算可见行数（至少 2 行）
    var visibleRows = Math.max(2, Math.floor(usableHeight / estimatedRowHeight));
    // 响应式调整：移动端减少行数
    if (window.innerWidth < 640) {
        visibleRows = Math.min(visibleRows, 3);
    } else if (window.innerWidth < 768) {
        visibleRows = Math.min(visibleRows, 4);
    }
    return visibleRows * cols;
}
