// 侧边栏开关（默认隐藏，localStorage 持久化）
(function() {
    const SIDEBAR_VISIBLE_KEY = 'fastpic_sidebar_visible';

    function isSidebarVisible() {
        return localStorage.getItem(SIDEBAR_VISIBLE_KEY) === 'true';
    }

    function applySidebarState(visible) {
        const sidebar = document.getElementById('sidebar');
        const handle = document.getElementById('resize-handle');
        const btn = document.getElementById('sidebar-toggle-btn');
        const overlay = document.getElementById('sidebar-overlay');
        if (visible) {
            sidebar.classList.remove('hidden');
            handle.classList.remove('hidden');
            if (overlay) overlay.classList.remove('hidden');
            btn.classList.add('bg-blue-50', 'text-blue-600');
            btn.classList.remove('text-slate-500');
        } else {
            sidebar.classList.add('hidden');
            handle.classList.add('hidden');
            if (overlay) overlay.classList.add('hidden');
            btn.classList.remove('bg-blue-50', 'text-blue-600');
            btn.classList.add('text-slate-500');
        }
    }

    // 页面加载时恢复状态
    applySidebarState(isSidebarVisible());

    // 全局切换函数
    window.toggleSidebar = function() {
        const visible = !isSidebarVisible();
        localStorage.setItem(SIDEBAR_VISIBLE_KEY, String(visible));
        applySidebarState(visible);
    };
})();

// ---------- gallery-top-bar 移入顶栏兜底：应对 htmx:afterSwap 未及时触发的极端情况 ----------
(function() {
    function ensureTopBarInSlot() {
        var container = document.getElementById('gallery-container');
        var slot = document.getElementById('gallery-top-slot');
        if (!container || !slot) return;
        var topBar = container.querySelector('.gallery-top-bar');
        if (topBar && !slot.contains(topBar)) {
            slot.innerHTML = '';
            slot.appendChild(topBar);
        }
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', ensureTopBarInSlot);
    } else {
        ensureTopBarInSlot();
    }
    // 延迟再检查一次，应对首屏 gallery 异步加载完成时 afterSwap 早于本脚本执行的情况
    setTimeout(ensureTopBarInSlot, 500);
})();

// ---------- galleryPathCache：LRU 缓存，TTL 7min，max 40 条 ----------
(function() {
    var MAX_ENTRIES = 40;
    var TTL_MS = 7 * 60 * 1000;  // 7 分钟
    var map = new Map();  // key (URL) -> { html, timestamp }

    window.galleryPathCache = {
        /** @param {string} key - 完整 gallery URL（与 buildGalleryUrl 一致） */
        get: function(key) {
            var entry = map.get(key);
            if (!entry) return null;
            if (Date.now() - entry.timestamp > TTL_MS) {
                map.delete(key);
                return null;
            }
            // LRU touch: 移到末尾
            map.delete(key);
            map.set(key, entry);
            return entry.html;
        },
        /** @param {string} key - 完整 gallery URL */
        set: function(key, html) {
            if (map.has(key)) map.delete(key);
            while (map.size >= MAX_ENTRIES) {
                map.delete(map.keys().next().value);
            }
            map.set(key, { html: html, timestamp: Date.now() });
        },
        /** 清空缓存（上传/删除/移动/重命名/新建文件夹后调用） */
        clear: function() {
            map.clear();
        }
    };
})();

// ---------- folderImagesCache：大图模式全量图片列表缓存 ----------
(function() {
    var cache = {};
    var TTL_MS = 5 * 60 * 1000;  // 5 分钟

    function buildFolderImagesKey(marker) {
        if (!marker) return null;
        var p = marker.getAttribute('data-path') || '';
        var s = marker.getAttribute('data-search') || '';
        var m = marker.getAttribute('data-mode') || 'folder';
        var sb = marker.getAttribute('data-sort-by') || 'modified_at';
        var so = marker.getAttribute('data-sort-order') || 'desc';
        var ff = marker.getAttribute('data-filter-filename') || '';
        var fsm = marker.getAttribute('data-filter-size-min') || '';
        var fsx = marker.getAttribute('data-filter-size-max') || '';
        var fdf = marker.getAttribute('data-filter-date-from') || '';
        var fdt = marker.getAttribute('data-filter-date-to') || '';
        var ft = marker.getAttribute('data-filter-tag') || '';
        return p + '|' + s + '|' + m + '|' + sb + '|' + so + '|' + ff + '|' + fsm + '|' + fsx + '|' + fdf + '|' + fdt + '|' + ft;
    }

    window.folderImagesCache = {
        get: function() {
            var marker = document.getElementById('current-path-marker');
            var key = buildFolderImagesKey(marker);
            if (!key || !cache[key]) return null;
            var entry = cache[key];
            if (Date.now() - entry.timestamp > TTL_MS) {
                delete cache[key];
                return null;
            }
            return entry.data;
        },
        set: function(data) {
            var marker = document.getElementById('current-path-marker');
            var key = buildFolderImagesKey(marker);
            if (!key) return;
            cache[key] = { data: data, timestamp: Date.now() };
        },
        clear: function() {
            cache = {};
        },
        prefetch: function() {
            var marker = document.getElementById('current-path-marker');
            if (!marker) return;
            var total = parseInt(marker.getAttribute('data-total') || '0', 10);
            if (total <= 0) return;
            var params = new URLSearchParams({
                path: marker.getAttribute('data-path') || '',
                search: marker.getAttribute('data-search') || '',
                mode: marker.getAttribute('data-mode') || 'folder',
                sort_by: marker.getAttribute('data-sort-by') || 'modified_at',
                sort_order: marker.getAttribute('data-sort-order') || 'desc',
                filter_filename: marker.getAttribute('data-filter-filename') || '',
                filter_size_min: marker.getAttribute('data-filter-size-min') || '',
                filter_size_max: marker.getAttribute('data-filter-size-max') || '',
                filter_date_from: marker.getAttribute('data-filter-date-from') || '',
                filter_date_to: marker.getAttribute('data-filter-date-to') || '',
                filter_tag: marker.getAttribute('data-filter-tag') || ''
            });
            fetch('/api/folder-images?' + params.toString())
                .then(function(r) { return r.json(); })
                .then(function(data) {
                    if (data.urls && data.ids) {
                        window.folderImagesCache.set(data);
                    }
                })
                .catch(function() {});
        }
    };
})();

/** 从 gallery 导航链接获取缓存键（完整 URL） */
function getGalleryCacheKeyFromLink(link) {
    var hxGet = link.getAttribute('hx-get');
    if (!hxGet || hxGet.indexOf('/gallery') === -1) return null;
    var target = link.getAttribute('hx-target');
    if (target && target !== '#gallery-container') return null;
    if (link.getAttribute('hx-include')) {
        try {
            var url = new URL(hxGet, location.origin);
            var params = url.searchParams;
            var path = params.get('path') || '';
            var opts = {};
            var filterTag = params.get('filter_tag') || '';
            var filterFilename = params.get('filter_filename') || '';
            var filterSizeMin = params.get('filter_size_min') || '';
            var filterSizeMax = params.get('filter_size_max') || '';
            var filterDateFrom = params.get('filter_date_from') || '';
            var filterDateTo = params.get('filter_date_to') || '';
            if (filterTag || filterFilename || filterSizeMin || filterSizeMax || filterDateFrom || filterDateTo) {
                opts.filters = {
                    filter_tag: filterTag,
                    filter_filename: filterFilename,
                    filter_size_min: filterSizeMin,
                    filter_size_max: filterSizeMax,
                    filter_date_from: filterDateFrom,
                    filter_date_to: filterDateTo
                };
            }
            if (params.get('mode')) opts.mode = params.get('mode');
            if (params.get('sort_by')) opts.sortBy = params.get('sort_by');
            if (params.get('sort_order')) opts.sortOrder = params.get('sort_order');
            if (params.get('cols')) opts.cols = params.get('cols');
            return typeof buildGalleryUrl === 'function' ? buildGalleryUrl(path, opts) : null;
        } catch (e) { return null; }
    }
    return hxGet;
}

/** 缓存命中时应用 gallery 内容并执行与 htmx:afterSwap 一致的后处理 */
function applyGalleryFromCache(html) {
    var container = document.getElementById('gallery-container');
    if (!container) return;
    container.innerHTML = html;
    if (typeof htmx !== 'undefined') htmx.process(container);
    var marker = document.getElementById('current-path-marker');
    var path = marker ? (marker.getAttribute('data-path') || '') : '';
    var newUrl = path ? '/?path=' + encodeURIComponent(path) : '/';
    var urlPath = getPathFromUrl();
    if (urlPath === path) {
        history.replaceState({path: path}, '', newUrl);
    } else {
        history.pushState({path: path}, '', newUrl);
    }
    var topBar = container.querySelector('.gallery-top-bar');
    var slot = document.getElementById('gallery-top-slot');
    if (topBar && slot) {
        slot.innerHTML = '';
        slot.appendChild(topBar);
    }
    var saved = localStorage.getItem('fastpic_gallery_cols');
    var sc = document.getElementById('scroll-container');
    if (sc && saved) sc.style.setProperty('--gallery-cols', saved);
    var mode = marker ? (marker.getAttribute('data-mode') || 'folder') : 'folder';
    var sortBy = marker ? (marker.getAttribute('data-sort-by') || 'modified_at') : 'modified_at';
    var sortOrder = marker ? (marker.getAttribute('data-sort-order') || 'desc') : 'desc';
    var pathInput = document.querySelector('[name=path]');
    if (pathInput) pathInput.value = path;
    var modeInput = document.getElementById('mode-input');
    if (modeInput) modeInput.value = mode;
    var sortByInput = document.getElementById('sort-by-input');
    if (sortByInput) sortByInput.value = sortBy;
    var sortOrderInput = document.getElementById('sort-order-input');
    if (sortOrderInput) sortOrderInput.value = sortOrder;
    if (marker && typeof window.getFilterState === 'function') {
        var filterState = window.getFilterState();
        filterState.filter_filename = marker.getAttribute('data-filter-filename') || '';
        filterState.filter_size_min = marker.getAttribute('data-filter-size-min') || '';
        filterState.filter_size_max = marker.getAttribute('data-filter-size-max') || '';
        filterState.filter_date_from = marker.getAttribute('data-filter-date-from') || '';
        filterState.filter_date_to = marker.getAttribute('data-filter-date-to') || '';
        filterState.filter_tag = marker.getAttribute('data-filter-tag') || '';
    }
    if (typeof window.syncGalleryGridCols === 'function') window.syncGalleryGridCols();
    container.dispatchEvent(new CustomEvent('htmx:afterSettle', { bubbles: true, detail: { target: container } }));
}

// ---------- gallery 导航链接点击委托：缓存命中时 preventDefault 并手动更新 DOM + history ----------
(function() {
    document.body.addEventListener('click', function(ev) {
        if (window._selectMode) return;
        var link = ev.target.closest('a[hx-get][hx-target="#gallery-container"]');
        if (!link) return;
        var cacheKey = getGalleryCacheKeyFromLink(link);
        if (!cacheKey || !window.galleryPathCache) return;
        var cached = window.galleryPathCache.get(cacheKey);
        if (!cached) return;
        ev.preventDefault();
        ev.stopPropagation();
        applyGalleryFromCache(cached);
    }, true);
})();

let modalImages = [];
let modalImageIds = [];
let modalMediaTypes = [];
let modalIndex = 0;
/** 大图切换请求 ID，用于校验加载完成时是否仍为当前请求，避免快速滑动时显示过时图片 */
var _modalRequestId = 0;
/** 图片信息内存缓存，避免重复 fetch /api/image-info */
var imageInfoCache = {};
let slideshowTimer = null;
let slideshowInterval = parseFloat(localStorage.getItem('fastpic_slideshow_interval') || '2');
let slideshowMode = localStorage.getItem('fastpic_slideshow_mode') || 'loop';
let videoAutoPlayNext = localStorage.getItem('fastpic_video_autoplay_next') === 'true';

var _mpegtsPlayer = null;
function _destroyMpegtsPlayer() {
    if (_mpegtsPlayer) {
        try { _mpegtsPlayer.destroy(); } catch (e) {}
        _mpegtsPlayer = null;
    }
}

/** 根据 URL 推断媒体类型（无 mediaTypes 时回退） */
function _inferMediaType(url) {
    if (!url) return 'image';
    var u = (url + '').toLowerCase();
    if (u.match(/\.(mp4|webm|mov|mkv|ts)(\?|$)/)) return 'video';
    return 'image';
}

/** 大图模式预加载前后各 N 张，提升左右切换流畅度（仅图片）
 * 桌面端预加载更多（3-4 张），移动端保持 2 张；根据滑动方向动态增加该侧预加载数量 */
function _getModalPreloadCounts(direction) {
    var isDesktop = window.innerWidth >= 768 && !('ontouchstart' in window);
    var base = isDesktop ? 6 : 4;
    var extra = isDesktop ? 2 : 1;  // 滑动方向侧额外预加载
    if (direction === 'prev') return { prev: base + extra, next: base };
    if (direction === 'next') return { prev: base, next: base + extra };
    return { prev: base, next: base };
}
function preloadModalImages(centerIndex, direction) {
    if (!modalImages.length) return;
    var types = modalMediaTypes.length ? modalMediaTypes : modalImages.map(_inferMediaType);
    var counts = _getModalPreloadCounts(direction || null);
    var start = Math.max(0, centerIndex - counts.prev);
    var end = Math.min(modalImages.length - 1, centerIndex + counts.next);
    for (var i = start; i <= end; i++) {
        if (i === centerIndex) continue;
        if (types[i] === 'image') {
            var img = new Image();
            img.src = modalImages[i];
        }
    }
}

function _showModalContent(url, mediaType) {
    var imgEl = document.getElementById('modal-img');
    var vidEl = document.getElementById('modal-video');
    if (!imgEl || !vidEl) return;
    var mt = mediaType || _inferMediaType(url);
    if (mt === 'video') {
        _destroyMpegtsPlayer();
        imgEl.classList.add('hidden');
        imgEl.removeAttribute('src');
        imgEl.classList.remove('modal-img-loading');
        vidEl.classList.remove('hidden');
        vidEl.removeAttribute('src');
        var vidFilename = (url && url.split('/').pop()) ? decodeURIComponent(url.split('/').pop().split('?')[0]) : '';
        vidEl.title = vidFilename || '视频预览';
        var isTs = /\.ts(\?|$)/i.test(url);
        var fullUrl = url.startsWith('/') ? (window.location.origin + url) : url;
        if (isTs && typeof mpegts !== 'undefined' && mpegts.isSupported()) {
            _mpegtsPlayer = mpegts.createPlayer({
                type: 'mpegts',
                isLive: false,
                url: fullUrl,
                cors: true
            }, { lazyLoad: true });
            _mpegtsPlayer.attachMediaElement(vidEl);
            _mpegtsPlayer.load();
            _mpegtsPlayer.play().catch(function() {
                vidEl.muted = true;
                vidEl.play().catch(function(){});
            });
        } else {
            vidEl.src = url;
            vidEl.load();
            var p = vidEl.play();
            if (p && typeof p.then === 'function') {
                p.catch(function() {
                    vidEl.muted = true;
                    vidEl.play().catch(function(){});
                });
            }
        }
    } else {
        _destroyMpegtsPlayer();
        vidEl.classList.add('hidden');
        vidEl.pause();
        vidEl.removeAttribute('src');
        imgEl.classList.remove('hidden');
        imgEl.classList.add('modal-img-loading');
        imgEl.src = url;
        imgEl.alt = (url && url.split('/').pop()) ? decodeURIComponent(url.split('/').pop()) : '图片预览';
        if (imgEl.complete && imgEl.naturalWidth) {
            imgEl.classList.remove('modal-img-loading');
        } else {
            imgEl.onload = function() {
                imgEl.classList.remove('modal-img-loading');
                imgEl.onload = null;
            };
        }
    }
}

/** 预加载完成后显示，避免切换时白屏。requestId 用于校验：加载完成时若已切换则丢弃，避免过时图片覆盖 */
function _showModalContentWhenReady(url, mediaType, requestId, cb) {
    var mt = mediaType || _inferMediaType(url);
    if (mt === 'video') {
        _showModalContent(url, mediaType);
        if (cb) cb();
    } else {
        var img = new Image();
        img.onload = img.onerror = function() {
            if (requestId !== _modalRequestId) return;
            _showModalContent(url, mediaType);
            if (cb) cb();
        };
        img.src = url;
    }
}

var _lastFocusedBeforeModal = null;

function openModal(photoUrl, index, allUrls, allIds, mediaTypes) {
    modalImages = allUrls || [];
    modalImageIds = allIds || [];
    modalMediaTypes = mediaTypes || modalImages.map(_inferMediaType);
    modalIndex = index;
    _showModalContent(photoUrl, modalMediaTypes[index]);
    var modal = document.getElementById('modal');
    _lastFocusedBeforeModal = document.activeElement;
    requestAnimationFrame(function() {
        requestAnimationFrame(function() {
            modal.classList.add('modal-open');
            modal.setAttribute('aria-hidden', 'false');
            var closeBtn = document.getElementById('modal-close-btn');
            if (closeBtn) closeBtn.focus();
        });
    });
    /* 应用工具栏展开/收起状态，保证移动端双行布局初始状态正确 */
    applyModalToolbarState(isModalToolbarCollapsed());
    _updateIntervalUI(slideshowInterval);
    _updateSlideshowModeUI(slideshowMode);
    _updateVideoAutoPlayNextUI();
    _bindVideoEndedHandler();
    _updateModalImageCounter();
    preloadModalImages(modalIndex);
}

function openModalFromGallery(photoUrl, index, pageUrls, pageIds, pageMediaTypes) {
    openModal(photoUrl, index, pageUrls, pageIds, pageMediaTypes);
    var marker = document.getElementById('current-path-marker');
    if (!marker) return;
    var total = parseInt(marker.getAttribute('data-total') || '0', 10);
    if (total <= pageUrls.length) return;

    // 优先使用预缓存的全量图片列表
    var cachedData = null;
    if (window.folderImagesCache) {
        cachedData = window.folderImagesCache.get();
    }
    if (cachedData && cachedData.urls && cachedData.ids && cachedData.ids.length > 0) {
        var currentId = modalImageIds[modalIndex];
        var newIndex = cachedData.ids.indexOf(currentId);
        if (newIndex < 0) newIndex = cachedData.urls.indexOf(modalImages[modalIndex]);
        if (newIndex < 0) newIndex = 0;
        modalImages = cachedData.urls;
        modalImageIds = cachedData.ids;
        modalMediaTypes = cachedData.media_types || modalImages.map(_inferMediaType);
        modalIndex = newIndex;
        _showModalContent(modalImages[modalIndex], modalMediaTypes[modalIndex]);
        _updateModalImageCounter();
        preloadModalImages(modalIndex);
        return;
    }

    // 缓存未命中，发起请求
    var params = new URLSearchParams({
        path: marker.getAttribute('data-path') || '',
        search: marker.getAttribute('data-search') || '',
        mode: marker.getAttribute('data-mode') || 'folder',
        sort_by: marker.getAttribute('data-sort-by') || 'modified_at',
        sort_order: marker.getAttribute('data-sort-order') || 'desc',
        filter_filename: marker.getAttribute('data-filter-filename') || '',
        filter_size_min: marker.getAttribute('data-filter-size-min') || '',
        filter_size_max: marker.getAttribute('data-filter-size-max') || '',
        filter_date_from: marker.getAttribute('data-filter-date-from') || '',
        filter_date_to: marker.getAttribute('data-filter-date-to') || '',
        filter_tag: marker.getAttribute('data-filter-tag') || ''
    });
    fetch('/api/folder-images?' + params.toString())
        .then(function(r) { return r.json(); })
        .then(function(data) {
            if (!data.urls || !data.ids || data.urls.length === 0) return;
            // 存入缓存供后续使用
            if (window.folderImagesCache) {
                window.folderImagesCache.set(data);
            }
            var currentId = modalImageIds[modalIndex];
            var newIndex = data.ids.indexOf(currentId);
            if (newIndex < 0) newIndex = data.urls.indexOf(modalImages[modalIndex]);
            if (newIndex < 0) newIndex = 0;
            modalImages = data.urls;
            modalImageIds = data.ids;
            modalMediaTypes = data.media_types || modalImages.map(_inferMediaType);
            modalIndex = newIndex;
            _showModalContent(modalImages[modalIndex], modalMediaTypes[modalIndex]);
            _updateModalImageCounter();
            preloadModalImages(modalIndex);
        })
        .catch(function(err) { console.error('加载全部媒体失败:', err); });
}
window.openModalFromGallery = openModalFromGallery;

function _doCloseModal() {
    stopSlideshow();
    _destroyMpegtsPlayer();
    var vid = document.getElementById('modal-video');
    if (vid) { vid.pause(); vid.removeAttribute('src'); }
    if (document.fullscreenElement) document.exitFullscreen();
    var modal = document.getElementById('modal');
    if (modal) {
        modal.classList.remove('modal-open');
        modal.setAttribute('aria-hidden', 'true');
    }
    var panel = document.getElementById('modal-image-info-panel');
    if (panel) panel.classList.add('hidden');
    _updateImageInfoBtnState(false);
    if (_lastFocusedBeforeModal && typeof _lastFocusedBeforeModal.focus === 'function') {
        _lastFocusedBeforeModal.focus();
    }
}

function closeModal(e) {
    if (e && (e.target.id === 'modal' || e.target.id === 'modal-overlay')) {
        _doCloseModal();
    }
}

function closeModalAndStopSlideshow() {
    _doCloseModal();
}

function handleModalContentClick(e) {
    if (e.target !== e.currentTarget) return;
    var isFullscreen = !!document.fullscreenElement;
    var isVideo = modalMediaTypes.length && modalMediaTypes[modalIndex] === 'video';
    if (isFullscreen && isVideo) {
        var rect = e.currentTarget.getBoundingClientRect();
        var w = rect.width;
        var x = e.clientX - rect.left;
        if (x < w * 0.25) {
            prevImage();
            return;
        }
        if (x > w * 0.75) {
            nextImage();
            return;
        }
    }
    closeModalAndStopSlideshow();
}

function _updateImageInfoBtnState(active) {
    var btn = document.getElementById('modal-info-btn');
    if (!btn) return;
    if (active) {
        btn.classList.add('bg-blue-500/60', 'text-white');
        btn.classList.remove('bg-white/15', 'text-white/80');
    } else {
        btn.classList.remove('bg-blue-500/60', 'text-white');
        btn.classList.add('bg-white/15', 'text-white/80');
    }
}

function toggleImageInfo() {
    var panel = document.getElementById('modal-image-info-panel');
    if (!panel) return;
    if (panel.classList.contains('hidden')) {
        showImageInfo();
    } else {
        panel.classList.add('hidden');
        _updateImageInfoBtnState(false);
    }
}

const MODAL_TOOLBAR_COLLAPSED_KEY = 'fastpic_modal_toolbar_collapsed';
function isModalToolbarCollapsed() {
    return localStorage.getItem(MODAL_TOOLBAR_COLLAPSED_KEY) !== 'false';
}
function setModalToolbarCollapsed(collapsed) {
    localStorage.setItem(MODAL_TOOLBAR_COLLAPSED_KEY, String(collapsed));
}
function applyModalToolbarState(collapsed) {
    var toolbar = document.getElementById('modal-toolbar');
    var wrap = document.getElementById('modal-toolbar-buttons');
    var expandIcon = document.getElementById('modal-toolbar-expand-icon');
    var collapseIcon = document.getElementById('modal-toolbar-collapse-icon');
    if (!wrap || !expandIcon || !collapseIcon) return;
    if (collapsed) {
        wrap.classList.add('max-w-0', 'opacity-0');
        wrap.classList.remove('max-w-[999px]', 'opacity-100');
        expandIcon.classList.remove('hidden');
        collapseIcon.classList.add('hidden');
        toolbar?.classList.remove('modal-toolbar-expanded');
    } else {
        wrap.classList.remove('max-w-0', 'opacity-0');
        wrap.classList.add('max-w-[999px]', 'opacity-100');
        expandIcon.classList.add('hidden');
        collapseIcon.classList.remove('hidden');
        toolbar?.classList.add('modal-toolbar-expanded');
    }
}
function toggleModalToolbar() {
    var collapsed = !isModalToolbarCollapsed();
    setModalToolbarCollapsed(collapsed);
    applyModalToolbarState(collapsed);
}

/** 将 image-info 数据渲染到 content 元素 */
function _renderImageInfoContent(content, data, imageId) {
    var tags = data.tags || [];
    var tagsHtml = '<div class="flex justify-between gap-3 mt-2 pt-2 border-t border-white/20"><span class="text-white/50 flex-shrink-0">标签</span><span class="flex flex-wrap gap-1 justify-end">';
    tags.forEach(function(t) {
        tagsHtml += '<span class="modal-tag-pill inline-flex items-center gap-1 px-2 py-0.5 rounded bg-white/20 hover:bg-red-500/60 cursor-pointer text-xs" data-tag="' + escapeAttr(t) + '" title="点击移除">#' + escapeHtml(t) + ' <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-width="2" stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"/></svg></span>';
    });
    tagsHtml += '</span></div>';
    tagsHtml += '<div class="mt-2 relative"><input type="text" id="modal-add-tag-input" placeholder="添加标签（点击输入框选择或输入后按 Enter）..." class="w-full px-2 py-1 text-sm bg-white/10 border border-white/20 rounded text-white placeholder-white/50 focus:outline-none focus:ring-1 focus:ring-white/50" data-image-id="' + imageId + '"><div id="modal-tag-suggestions" class="hidden absolute left-0 right-0 top-full z-20 mt-1 max-h-48 overflow-y-auto bg-slate-800 rounded border border-white/10 text-sm shadow-lg"></div></div>';

    content.innerHTML =
        '<div class="flex justify-between gap-3"><span class="text-white/50 flex-shrink-0">完整路径</span><span class="break-all text-right">' + escapeHtml(data.full_path) + '</span></div>' +
        '<div class="flex justify-between gap-3"><span class="text-white/50 flex-shrink-0">文件名</span><span class="break-all text-right">' + escapeHtml(data.filename) + '</span></div>' +
        '<div class="flex justify-between gap-3"><span class="text-white/50 flex-shrink-0">分辨率</span><span>' + escapeHtml(data.resolution) + '</span></div>' +
        '<div class="flex justify-between gap-3"><span class="text-white/50 flex-shrink-0">文件大小</span><span>' + escapeHtml(data.file_size) + '</span></div>' +
        '<div class="flex justify-between gap-3"><span class="text-white/50 flex-shrink-0">修改时间</span><span>' + escapeHtml(data.modified_at) + '</span></div>' +
        tagsHtml;

    _bindModalTagHandlers(content, imageId);
}

/** 使指定图片的信息缓存失效（如标签修改后） */
function invalidateImageInfoCache(imageId) {
    if (imageId != null) delete imageInfoCache[imageId];
}

/** 加载并显示当前图片信息到左下角面板（不负责开关面板，仅填充内容并显示） */
function showImageInfo() {
    var panel = document.getElementById('modal-image-info-panel');
    var content = document.getElementById('image-info-content');
    if (!panel || !content) return;

    if (modalImageIds.length === 0 || modalIndex < 0 || modalIndex >= modalImageIds.length) {
        content.innerHTML = '<div class="text-white/70">暂无媒体信息</div>';
        panel.classList.remove('hidden');
        _updateImageInfoBtnState(true);
        return;
    }
    var imageId = modalImageIds[modalIndex];

    var cached = imageInfoCache[imageId];
    if (cached) {
        panel.classList.remove('hidden');
        _updateImageInfoBtnState(true);
        _renderImageInfoContent(content, cached, imageId);
        return;
    }

    content.innerHTML = '<div class="text-white/50">加载中...</div>';
    panel.classList.remove('hidden');
    _updateImageInfoBtnState(true);

    fetch('/api/image-info/' + imageId)
        .then(function(res) {
            if (!res.ok) {
                if (res.status === 404) return Promise.reject('媒体文件不存在或已被删除');
                return Promise.reject('获取信息失败');
            }
            return res.json();
        })
        .then(function(data) {
            imageInfoCache[imageId] = data;
            _renderImageInfoContent(content, data, imageId);
        })
        .catch(function(err) {
            content.innerHTML = '<div class="text-red-300">' + escapeHtml(String(err)) + '</div>';
        });
}

function _bindModalTagHandlers(container, imageId) {
    if (!container) return;
    container.addEventListener('click', function(e) {
        var pill = e.target.closest('.modal-tag-pill');
        if (pill) {
            e.preventDefault();
            e.stopPropagation();
            var tag = pill.getAttribute('data-tag');
            if (!tag) return;
            fetch('/api/images/' + imageId + '/tags/' + encodeURIComponent(tag), { method: 'DELETE' })
                .then(function(r) { return r.json(); })
                .then(function() {
                    invalidateImageInfoCache(imageId);
                    showImageInfo();
                });
        }
    });
    var addInput = container.querySelector('#modal-add-tag-input');
    var suggestions = container.querySelector('#modal-tag-suggestions');
    if (addInput && suggestions) {
        var allTagsCache = [];
        function renderTagButtons(tags) {
            suggestions.innerHTML = '';
            tags.forEach(function(t) {
                var btn = document.createElement('button');
                btn.type = 'button';
                btn.className = 'block w-full text-left px-3 py-2 hover:bg-white/10 text-white/90';
                btn.textContent = '#' + t.name + (t.count > 0 ? ' (' + t.count + ')' : '');
                btn.addEventListener('click', function() {
                    fetch('/api/images/' + imageId + '/tags', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ tags: [t.name] })
                    }).then(function(r) { return r.json(); }).then(function() {
                        addInput.value = '';
                        suggestions.classList.add('hidden');
                        invalidateImageInfoCache(imageId);
                        showImageInfo();
                    });
                });
                suggestions.appendChild(btn);
            });
            suggestions.classList.toggle('hidden', !suggestions.children.length);
            suggestions.style.minWidth = addInput.offsetWidth + 'px';
        }
        function loadAndShowTags(q) {
            if (allTagsCache.length > 0) {
                var filtered = q ? allTagsCache.filter(function(t) { return t.name.toLowerCase().indexOf(q.toLowerCase()) >= 0; }) : allTagsCache;
                renderTagButtons(filtered);
            } else {
                fetch('/api/tags?limit=200')
                    .then(function(r) { return r.json(); })
                    .then(function(data) {
                        allTagsCache = data.tags || [];
                        var filtered = q ? allTagsCache.filter(function(t) { return t.name.toLowerCase().indexOf(q.toLowerCase()) >= 0; }) : allTagsCache;
                        renderTagButtons(filtered);
                    });
            }
        }
        addInput.addEventListener('focus', function() {
            loadAndShowTags(addInput.value.trim());
        });
        addInput.addEventListener('input', function() {
            var q = addInput.value.trim();
            loadAndShowTags(q);
        });
        addInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                var tag = addInput.value.trim().replace(/^#+/, '');
                if (tag) {
                    fetch('/api/images/' + imageId + '/tags', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ tags: [tag] })
                    }).then(function(r) { return r.json(); }).then(function() {
                        addInput.value = '';
                        if (suggestions) suggestions.classList.add('hidden');
                        invalidateImageInfoCache(imageId);
                        showImageInfo();
                    });
                }
            }
        });
        var closeListenerActive = false;
        function setupCloseOnClickOutside() {
            if (closeListenerActive) return;
            closeListenerActive = true;
            function closeSuggestions(ev) {
                if (!addInput.contains(ev.target) && !suggestions.contains(ev.target)) {
                    suggestions.classList.add('hidden');
                    closeListenerActive = false;
                    document.removeEventListener('click', closeSuggestions);
                }
            }
            setTimeout(function() { document.addEventListener('click', closeSuggestions); }, 0);
        }
        var origRenderTagButtons = renderTagButtons;
        renderTagButtons = function(tags) {
            origRenderTagButtons(tags);
            if (tags.length > 0) setupCloseOnClickOutside();
        };
    }
}

function prevImage() {
    if (modalImages.length === 0) return;
    modalIndex = (modalIndex - 1 + modalImages.length) % modalImages.length;
    _modalRequestId++;
    var reqId = _modalRequestId;
    _showModalContentWhenReady(modalImages[modalIndex], modalMediaTypes[modalIndex], reqId, function() {
        _updateModalImageCounter();
        refreshImageInfoIfVisible();
        preloadModalImages(modalIndex, 'prev');
    });
}

function nextImage(fromSlideshow) {
    if (modalImages.length === 0) return false;
    var types = modalMediaTypes.length ? modalMediaTypes : modalImages.map(_inferMediaType);

    if (fromSlideshow) {
        // 幻灯片模式：跳过视频，只播图片
        var nextIdx = (modalIndex + 1) % modalImages.length;
        var visited = 0;
        while (types[nextIdx] === 'video' && visited < modalImages.length) {
            nextIdx = (nextIdx + 1) % modalImages.length;
            visited++;
        }
        if (visited >= modalImages.length || types[nextIdx] === 'video') {
            stopSlideshow();
            return false;
        }
        if (nextIdx <= modalIndex && slideshowMode === 'stop') {
            stopSlideshow();
            return false;
        }
        modalIndex = nextIdx;
    } else {
        if (modalIndex === modalImages.length - 1) {
            showConfirm('已经是最后一张，是否从头开始？', '从头开始', function() {
                modalIndex = 0;
                _modalRequestId++;
                var reqId = _modalRequestId;
                _showModalContentWhenReady(modalImages[modalIndex], modalMediaTypes[modalIndex], reqId, function() {
                    _updateModalImageCounter();
                    refreshImageInfoIfVisible();
                    preloadModalImages(modalIndex, 'next');
                });
            });
            return false;
        }
        modalIndex = modalIndex + 1;
    }
    _modalRequestId++;
    var reqId = _modalRequestId;
    _showModalContentWhenReady(modalImages[modalIndex], modalMediaTypes[modalIndex], reqId, function() {
        _updateModalImageCounter();
        refreshImageInfoIfVisible();
        preloadModalImages(modalIndex, 'next');
    });
    return true;
}

function refreshImageInfoIfVisible() {
    var panel = document.getElementById('modal-image-info-panel');
    if (panel && !panel.classList.contains('hidden')) {
        showImageInfo();
    }
}

// ── 幻灯片播放 ──

function toggleSlideshow() {
    if (slideshowTimer) {
        stopSlideshow();
    } else {
        startSlideshow();
    }
}

function startSlideshow() {
    if (modalImages.length <= 1) return;
    if (slideshowTimer) return;
    var types = modalMediaTypes.length ? modalMediaTypes : modalImages.map(_inferMediaType);
    if (types[modalIndex] === 'video') {
        if (!nextImage(true)) return;
    }
    slideshowTimer = setInterval(function() {
        nextImage(true);
    }, slideshowInterval * 1000);
    // 更新按钮外观
    var playIcon = document.getElementById('slideshow-play-icon');
    var pauseIcon = document.getElementById('slideshow-pause-icon');
    var btn = document.getElementById('slideshow-btn');
    if (playIcon) playIcon.classList.add('hidden');
    if (pauseIcon) pauseIcon.classList.remove('hidden');
    if (btn) { btn.classList.add('bg-blue-500/60'); btn.classList.remove('bg-white/15'); }
}

function stopSlideshow() {
    if (slideshowTimer) {
        clearInterval(slideshowTimer);
        slideshowTimer = null;
    }
    var playIcon = document.getElementById('slideshow-play-icon');
    var pauseIcon = document.getElementById('slideshow-pause-icon');
    var btn = document.getElementById('slideshow-btn');
    if (playIcon) playIcon.classList.remove('hidden');
    if (pauseIcon) pauseIcon.classList.add('hidden');
    if (btn) { btn.classList.remove('bg-blue-500/60'); btn.classList.add('bg-white/15'); }
}

function toggleSlideshowInterval() {
    var popover = document.getElementById('slideshow-interval-popover');
    if (popover) popover.classList.toggle('hidden');
}

function updateSlideshowInterval(val) {
    slideshowInterval = parseFloat(val);
    localStorage.setItem('fastpic_slideshow_interval', String(slideshowInterval));
    _updateIntervalUI(slideshowInterval);
    // 如果正在播放，重启计时器以使用新间隔
    if (slideshowTimer) {
        stopSlideshow();
        startSlideshow();
    }
}

function _updateIntervalUI(val) {
    var label = document.getElementById('slideshow-interval-label');
    var valueText = document.getElementById('slideshow-interval-value');
    var slider = document.getElementById('slideshow-interval-slider');
    var display = val % 1 === 0 ? val + ' 秒' : val.toFixed(1) + ' 秒';
    if (label) label.textContent = val % 1 === 0 ? val + 's' : val.toFixed(1) + 's';
    if (valueText) valueText.textContent = display;
    if (slider) slider.value = val;
}

function _updateSlideshowModeUI(mode) {
    var btns = document.querySelectorAll('.slideshow-mode-btn');
    btns.forEach(function(btn) {
        if (btn.getAttribute('data-mode') === mode) {
            btn.classList.add('bg-blue-500/60', 'text-white');
            btn.classList.remove('bg-white/15', 'text-white/80');
        } else {
            btn.classList.remove('bg-blue-500/60', 'text-white');
            btn.classList.add('bg-white/15', 'text-white/80');
        }
    });
}

function setSlideshowMode(mode) {
    slideshowMode = mode;
    localStorage.setItem('fastpic_slideshow_mode', mode);
    _updateSlideshowModeUI(mode);
}

// ── 视频自动播放下一个 ──
var _videoEndedHandlerBound = false;
function _bindVideoEndedHandler() {
    if (_videoEndedHandlerBound) return;
    _videoEndedHandlerBound = true;
    var vidEl = document.getElementById('modal-video');
    if (vidEl) {
        vidEl.addEventListener('ended', function() {
            if (videoAutoPlayNext && modalMediaTypes[modalIndex] === 'video') {
                nextImage();
            }
        });
    }
}

function toggleVideoAutoPlayNext() {
    videoAutoPlayNext = !videoAutoPlayNext;
    localStorage.setItem('fastpic_video_autoplay_next', String(videoAutoPlayNext));
    _updateVideoAutoPlayNextUI();
}

function _updateVideoAutoPlayNextUI() {
    var btn = document.getElementById('video-autoplay-next-btn');
    if (!btn) return;
    if (videoAutoPlayNext) {
        btn.classList.add('bg-blue-500/60', 'text-white');
        btn.classList.remove('bg-white/15', 'text-white/80');
    } else {
        btn.classList.remove('bg-blue-500/60', 'text-white');
        btn.classList.add('bg-white/15', 'text-white/80');
    }
}

function _updateModalImageCounter() {
    var el = document.getElementById('modal-image-counter');
    if (!el) return;
    var total = modalImages.length;
    var current = modalIndex + 1;
    el.textContent = current + '/' + total;
}

// ── 全屏查看 ──

function toggleModalFullscreen() {
    var modal = document.getElementById('modal');
    if (!document.fullscreenElement) {
        modal.requestFullscreen().catch(function(err) {
            console.warn('无法进入全屏:', err);
        });
    } else {
        document.exitFullscreen();
    }
}

function _updateFullscreenUI() {
    var enterIcon = document.getElementById('fullscreen-enter-icon');
    var exitIcon = document.getElementById('fullscreen-exit-icon');
    var btn = document.getElementById('modal-fullscreen-btn');
    if (document.fullscreenElement) {
        if (enterIcon) enterIcon.classList.add('hidden');
        if (exitIcon) exitIcon.classList.remove('hidden');
        if (btn) btn.title = '退出全屏 (F)';
    } else {
        if (enterIcon) enterIcon.classList.remove('hidden');
        if (exitIcon) exitIcon.classList.add('hidden');
        if (btn) btn.title = '全屏查看 (F)';
    }
}

document.addEventListener('fullscreenchange', _updateFullscreenUI);

// 点击模态框外部区域关闭间隔弹窗
document.addEventListener('click', function(e) {
    var popover = document.getElementById('slideshow-interval-popover');
    var wrapper = document.getElementById('slideshow-interval-wrapper');
    if (popover && !popover.classList.contains('hidden') && wrapper && !wrapper.contains(e.target)) {
        popover.classList.add('hidden');
    }
});

function handleSetThumbnailOption(action) {
    closeSetThumbnailPopover();
    if (action === 'parent') {
        if (window.setAsFolderThumbnail) window.setAsFolderThumbnail();
    } else if (action === 'browse') {
        if (window.showSetThumbnailFolderDialog) window.showSetThumbnailFolderDialog();
    }
}
window.handleSetThumbnailOption = handleSetThumbnailOption;

// 设为缩略图选单：点击外部关闭
document.addEventListener('click', function(e) {
    var setThumbPopover = document.getElementById('set-thumbnail-popover');
    var setThumbWrapper = document.getElementById('modal-set-folder-thumb-wrapper');
    if (setThumbPopover && !setThumbPopover.classList.contains('hidden') && setThumbWrapper && !setThumbWrapper.contains(e.target)) {
        closeSetThumbnailPopover();
    }
});

function deleteCurrentImage() {
    if (modalImages.length === 0 || modalImageIds.length === 0) return;
    var imageId = modalImageIds[modalIndex];
    if (!imageId) return;

    showConfirm('确定要删除当前媒体文件吗？此操作不可恢复。', '删除', function() {
        var btn = document.getElementById('modal-delete-btn');
        if (btn) btn.disabled = true;

        fetch('/api/delete-images', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ids: [imageId]})
        }).then(function(r) { return r.json(); }).then(function(data) {
            // 从列表中移除当前项
            modalImages.splice(modalIndex, 1);
            modalImageIds.splice(modalIndex, 1);
            modalMediaTypes.splice(modalIndex, 1);

            if (modalImages.length === 0) {
                // 没有媒体了，关闭模态框并刷新画廊
                closeModalAndStopSlideshow();
                refreshGalleryFromModal();
            } else {
                // 调整索引，显示下一张（或最后一张）
                if (modalIndex >= modalImages.length) {
                    modalIndex = modalImages.length - 1;
                }
                _showModalContent(modalImages[modalIndex], modalMediaTypes[modalIndex]);
                _updateModalImageCounter();
                refreshImageInfoIfVisible();
                preloadModalImages(modalIndex);
                // 刷新画廊（后台更新网格）
                refreshGalleryFromModal();
            }
            if (btn) btn.disabled = false;
        }).catch(function(err) {
            console.error('删除失败:', err);
            if (typeof showToast === 'function') showToast('删除失败', 'error');
            if (btn) btn.disabled = false;
        });
    }, null, { variant: 'danger' });
}

function downloadCurrentImage() {
    if (modalImages.length === 0) return;
    var imageId = modalImageIds[modalIndex];
    if (imageId != null) {
        window.location.href = '/api/download/image?id=' + encodeURIComponent(imageId);
    } else {
        var url = modalImages[modalIndex];
        if (url && url.indexOf('/photos/') === 0) {
            var rel = decodeURIComponent(url.slice('/photos/'.length).split('?')[0]);
            window.location.href = '/api/download/image?relative_path=' + encodeURIComponent(rel);
        }
    }
}

function setAsFolderThumbnail() {
    if (modalImages.length === 0) return;
    var url = modalImages[modalIndex];
    if (!url || url.indexOf('/photos/') !== 0) return;
    var relPath = decodeURIComponent(url.slice('/photos/'.length).split('?')[0]);
    var lastSlash = relPath.lastIndexOf('/');
    if (lastSlash < 0) {
        if (typeof showToast === 'function') showToast('根目录媒体文件无法设为文件夹缩略图', 'error');
        return;
    }
    var folderPath = relPath.slice(0, lastSlash);
    var apiUrl = '/api/folders/' + folderPath.split('/').map(encodeURIComponent).join('/') + '/thumbnails';
    fetch(apiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ relative_path: relPath })
    }).then(function(r) {
        if (r.ok) {
            if (typeof showToast === 'function') showToast('已设为文件夹缩略图');
            if (typeof refreshGalleryFromModal === 'function') refreshGalleryFromModal();
        } else {
            return r.json().then(function(d) {
                var msg = (d && d.detail) ? (typeof d.detail === 'string' ? d.detail : JSON.stringify(d.detail)) : '请求失败';
                if (typeof showToast === 'function') showToast(msg, 'error');
            }).catch(function() {
                if (typeof showToast === 'function') showToast('请求失败', 'error');
            });
        }
    }).catch(function(err) {
        if (typeof showToast === 'function') showToast('请求失败', 'error');
    });
}
window.setAsFolderThumbnail = setAsFolderThumbnail;

function toggleSetThumbnailPopover() {
    var popover = document.getElementById('set-thumbnail-popover');
    var btn = document.getElementById('modal-set-folder-thumb-btn');
    if (!popover || !btn) return;
    if (popover.classList.contains('hidden')) {
        var rect = btn.getBoundingClientRect();
        var popH = popover.scrollHeight || 80;
        var gap = 8;
        if (rect.bottom + gap + popH <= window.innerHeight - 8) {
            popover.style.top = (rect.bottom + gap) + 'px';
        } else {
            popover.style.top = (rect.top - popH - gap) + 'px';
        }
        var minLeft = 16;
        var popoverWidth = 180;
        if (rect.right - popoverWidth < minLeft) {
            popover.style.left = minLeft + 'px';
            popover.style.right = 'auto';
        } else {
            popover.style.left = '';
            popover.style.right = (window.innerWidth - rect.right) + 'px';
        }
    }
    popover.classList.toggle('hidden');
}

function closeSetThumbnailPopover() {
    var popover = document.getElementById('set-thumbnail-popover');
    if (popover) popover.classList.add('hidden');
}

window.toggleSetThumbnailPopover = toggleSetThumbnailPopover;

function showSetThumbnailFolderDialog() {
    if (modalImages.length === 0) return;
    var url = modalImages[modalIndex];
    if (!url || url.indexOf('/photos/') !== 0) return;
    var relPath = decodeURIComponent(url.slice('/photos/'.length).split('?')[0]);
    var lastSlash = relPath.lastIndexOf('/');
    if (lastSlash < 0) {
        if (typeof showToast === 'function') showToast('根目录媒体文件无法设为文件夹缩略图', 'error');
        return;
    }
    var initialPath = relPath.slice(0, lastSlash);

    var old = document.getElementById('set-thumbnail-dialog');
    if (old) old.remove();

    var overlay = document.createElement('div');
    overlay.id = 'set-thumbnail-dialog';
    overlay.className = 'fixed inset-0 z-[100] flex items-center justify-center bg-black/40';
    overlay.innerHTML =
        '<div class="bg-white rounded-xl shadow-2xl max-w-md w-full mx-4 max-h-[85vh] flex flex-col">' +
            '<div class="p-4 border-b border-slate-200 flex-shrink-0">' +
                '<div class="flex items-center gap-3 mb-2">' +
                    '<div class="w-10 h-10 rounded-full bg-blue-100 flex items-center justify-center flex-shrink-0">' +
                        '<svg class="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-width="2" stroke-linecap="round" stroke-linejoin="round" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"/></svg>' +
                    '</div>' +
                    '<h3 class="text-lg font-semibold text-slate-800">设为文件夹缩略图</h3>' +
                '</div>' +
                '<div class="text-sm text-slate-500">选择要设置缩略图的文件夹</div>' +
                '<nav id="set-thumb-breadcrumb" class="mt-2 flex items-center gap-1 text-sm text-slate-600 flex-wrap"></nav>' +
            '</div>' +
            '<div id="set-thumb-folder-list" class="flex-1 overflow-y-auto p-4 min-h-0"></div>' +
            '<div class="p-4 border-t border-slate-200 flex-shrink-0 flex justify-end gap-2">' +
                '<button type="button" class="set-thumb-cancel-btn px-4 py-2 rounded-lg border border-slate-300 text-slate-700 hover:bg-slate-50 text-sm font-medium transition-colors">取消</button>' +
                '<button type="button" class="set-thumb-confirm-btn px-4 py-2 rounded-lg bg-blue-500 hover:bg-blue-600 text-white text-sm font-medium transition-colors">设为该文件夹缩略图</button>' +
            '</div>' +
            '<div id="set-thumb-error" class="hidden px-4 pb-4 text-sm text-red-500"></div>' +
        '</div>';

    document.body.appendChild(overlay);

    var breadcrumbEl = overlay.querySelector('#set-thumb-breadcrumb');
    var listEl = overlay.querySelector('#set-thumb-folder-list');
    var confirmBtn = overlay.querySelector('.set-thumb-confirm-btn');
    var errorEl = overlay.querySelector('#set-thumb-error');

    var browser = createFolderBrowser(breadcrumbEl, listEl, {
        initialPath: initialPath,
        itemClass: 'set-thumb-folder-item',
        navClass: 'set-thumb-nav-link'
    });
    browser.init();

    function doSetThumbnail() {
        errorEl.classList.add('hidden');
        var currentPath = browser.getPath();
        if (!currentPath) {
            if (typeof showToast === 'function') showToast('请选择具体文件夹', 'error');
            return;
        }
        confirmBtn.disabled = true;
        confirmBtn.textContent = '设置中...';
        var apiUrl = '/api/folders/' + currentPath.split('/').map(encodeURIComponent).join('/') + '/thumbnails';
        fetch(apiUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ relative_path: relPath })
        }).then(function(r) {
            overlay.remove();
            if (r.ok) {
                if (typeof showToast === 'function') showToast('已设为文件夹缩略图');
                if (typeof refreshGalleryFromModal === 'function') refreshGalleryFromModal();
            } else {
                return r.json().then(function(d) {
                    var msg = (d && d.detail) ? (typeof d.detail === 'string' ? d.detail : JSON.stringify(d.detail)) : '请求失败';
                    if (typeof showToast === 'function') showToast(msg, 'error');
                }).catch(function() {
                    if (typeof showToast === 'function') showToast('请求失败', 'error');
                });
            }
        }).catch(function(err) {
            if (typeof showToast === 'function') showToast('请求失败', 'error');
            confirmBtn.disabled = false;
            confirmBtn.textContent = '设为该文件夹缩略图';
        });
    }

    overlay.querySelector('.set-thumb-cancel-btn').addEventListener('click', function() { overlay.remove(); });
    confirmBtn.addEventListener('click', doSetThumbnail);
    overlay.addEventListener('click', function(e) { if (e.target === overlay) overlay.remove(); });
}
window.showSetThumbnailFolderDialog = showSetThumbnailFolderDialog;

function refreshGalleryFromModal() {
    if (window.galleryPathCache) window.galleryPathCache.clear();
    if (window.folderImagesCache) window.folderImagesCache.clear();
    var marker = document.getElementById('current-path-marker');
    var path = marker ? (marker.getAttribute('data-path') || '') : '';
    var opts = marker ? {
        mode: marker.getAttribute('data-mode') || 'folder',
        sortBy: marker.getAttribute('data-sort-by') || 'modified_at',
        sortOrder: marker.getAttribute('data-sort-order') || 'desc',
        cols: marker.getAttribute('data-cols') || (document.getElementById('cols-input') ? document.getElementById('cols-input').value : '4')
    } : undefined;
    htmx.ajax('GET', buildGalleryUrl(path, opts), {target: '#gallery-container', swap: 'innerHTML'});
}

document.addEventListener('keydown', function(e) {
    const modal = document.getElementById('modal');
    if (!modal || !modal.classList.contains('modal-open')) return;
    var active = document.activeElement;
    var isEditable = active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA' || active.isContentEditable);
    if (e.key === 'Escape') {
        if (document.fullscreenElement) {
            document.exitFullscreen();
        } else {
            closeModalAndStopSlideshow();
        }
    }
    if (e.key === 'ArrowLeft' && !isEditable) {
        var isFullscreen = !!document.fullscreenElement;
        var isVideo = modalMediaTypes.length && modalMediaTypes[modalIndex] === 'video';
        if (isFullscreen && isVideo) {
            var vid = document.getElementById('modal-video');
            if (vid && isFinite(vid.duration)) {
                e.preventDefault();
                vid.currentTime = Math.max(0, vid.currentTime - 5);
            }
        } else { stopSlideshow(); prevImage(); }
    }
    if (e.key === 'ArrowRight' && !isEditable) {
        var isFullscreen = !!document.fullscreenElement;
        var isVideo = modalMediaTypes.length && modalMediaTypes[modalIndex] === 'video';
        if (isFullscreen && isVideo) {
            var vid = document.getElementById('modal-video');
            if (vid && isFinite(vid.duration)) {
                e.preventDefault();
                vid.currentTime = Math.min(vid.duration, vid.currentTime + 5);
            }
        } else { stopSlideshow(); nextImage(); }
    }
    if ((e.key === 'Delete' || e.key === 'Backspace') && !isEditable) { deleteCurrentImage(); }
    if (e.key === ' ' && !isEditable) { e.preventDefault(); toggleSlideshow(); }
    if ((e.key === 'f' || e.key === 'F') && !isEditable) { toggleModalFullscreen(); }
});

// 大图模式：触摸滑动切换图片
(function() {
    var overlay = document.getElementById('modal-overlay');
    if (!overlay) return;
    var startX = 0, startY = 0;
    var SWIPE_THRESHOLD = 50;
    overlay.addEventListener('touchstart', function(e) {
        if (e.touches.length !== 1) return;
        startX = e.touches[0].clientX;
        startY = e.touches[0].clientY;
    }, {passive: true});
    overlay.addEventListener('touchend', function(e) {
        if (e.changedTouches.length !== 1) return;
        var endX = e.changedTouches[0].clientX;
        var endY = e.changedTouches[0].clientY;
        var dx = endX - startX;
        var dy = endY - startY;
        if (Math.abs(dx) < SWIPE_THRESHOLD) return;
        if (Math.abs(dx) < Math.abs(dy)) return;
        var isFullscreen = !!document.fullscreenElement;
        var isVideo = modalMediaTypes.length && modalMediaTypes[modalIndex] === 'video';
        if (isFullscreen && isVideo) {
            var vid = document.getElementById('modal-video');
            if (vid && isFinite(vid.duration) && vid.duration > 0) {
                var screenW = window.innerWidth;
                var seekDelta = (dx / screenW) * vid.duration;
                var newTime = Math.max(0, Math.min(vid.duration, vid.currentTime + seekDelta));
                vid.currentTime = newTime;
            }
        } else {
            stopSlideshow();
            if (dx > 0) prevImage();
            else nextImage();
        }
    }, {passive: true});
})();

// 大图模式：鼠标滚轮切换图片
(function() {
    var overlay = document.getElementById('modal-overlay');
    if (!overlay) return;
    overlay.addEventListener('wheel', function(e) {
        var modal = document.getElementById('modal');
        if (!modal.classList.contains('modal-open')) return;
        if (modalImages.length <= 1) return;
        var isFullscreen = !!document.fullscreenElement;
        var isVideo = modalMediaTypes.length && modalMediaTypes[modalIndex] === 'video';
        if (isFullscreen && isVideo) {
            var vid = document.getElementById('modal-video');
            if (vid && isFinite(vid.duration) && vid.duration > 0) {
                e.preventDefault();
                var seekDelta = (e.deltaY > 0 ? 1 : -1) * 3;
                vid.currentTime = Math.max(0, Math.min(vid.duration, vid.currentTime + seekDelta));
            }
            return;
        }
        e.preventDefault();
        stopSlideshow();
        if (e.deltaY > 0) nextImage();
        else prevImage();
    }, {passive: false});
})();

// 缩略图大小调节
(function() {
    const STORAGE_KEY = 'fastpic_gallery_cols';
    const DEFAULT_COLS = 4;
    const MIN_COLS = 2;
    const MAX_COLS = 8;

    function getScrollContainer() {
        return document.getElementById('scroll-container');
    }

    /** 按视口宽度限制最大列数（初始化与 resize 时使用） */
    function getMaxColsForViewport() {
        var w = window.innerWidth;
        if (w < 640) return 2;
        if (w < 768) return 3;
        if (w < 1024) return 4;
        if (w < 1280) return 6;
        return MAX_COLS;
    }

    function initThumbnailSize() {
        const saved = localStorage.getItem(STORAGE_KEY);
        const userCols = saved ? Math.max(MIN_COLS, Math.min(MAX_COLS, parseInt(saved, 10))) : DEFAULT_COLS;
        const cols = Math.min(userCols, getMaxColsForViewport());
        const sc = getScrollContainer();
        if (sc) sc.style.setProperty('--gallery-cols', String(cols));
        var colsInput = document.getElementById('cols-input');
        if (colsInput) colsInput.value = String(cols);
    }

    function setThumbnailCols(cols, skipRefresh) {
        const c = Math.max(MIN_COLS, Math.min(MAX_COLS, cols));
        const sc = getScrollContainer();
        if (sc) sc.style.setProperty('--gallery-cols', String(c));
        localStorage.setItem(STORAGE_KEY, String(c));
        var colsInput = document.getElementById('cols-input');
        if (colsInput) colsInput.value = String(c);
        const valEl = document.getElementById('thumbnail-size-value');
        if (valEl) valEl.textContent = c + ' 列';
        const slider = document.getElementById('thumbnail-size-slider');
        if (slider) slider.value = c;
        syncGalleryGridCols();
        if (!skipRefresh) {
            // 列数变化时重新加载画廊，使每页数量能被列数整除
            if (typeof window.refreshGallery === 'function') window.refreshGallery();
            // 瀑布流模式下重建列布局
            if (typeof window.rebuildWaterfall === 'function') window.rebuildWaterfall();
        }
    }

    function syncGalleryGridCols() {
        var grid = document.getElementById('gallery-grid');
        var sc = getScrollContainer();
        if (grid && sc) {
            var cols = getComputedStyle(sc).getPropertyValue('--gallery-cols').trim() || '4';
            grid.setAttribute('data-cols', cols);
        }
    }
    window.syncGalleryGridCols = syncGalleryGridCols;

    initThumbnailSize();

    // 窗口 resize 时按视口重新限制列数（不覆盖用户设置）
    var colsResizeTimer;
    window.addEventListener('resize', function() {
        clearTimeout(colsResizeTimer);
        colsResizeTimer = setTimeout(function() {
            var saved = localStorage.getItem(STORAGE_KEY);
            var userCols = saved ? Math.max(MIN_COLS, Math.min(MAX_COLS, parseInt(saved, 10))) : DEFAULT_COLS;
            var cols = Math.min(userCols, getMaxColsForViewport());
            var sc = getScrollContainer();
            if (sc) sc.style.setProperty('--gallery-cols', String(cols));
            var colsInput = document.getElementById('cols-input');
            if (colsInput) colsInput.value = String(cols);
            syncGalleryGridCols();
            if (typeof window.rebuildWaterfall === 'function') window.rebuildWaterfall();
        }, 150);
    });

    // 排序设定持久化
    (function initSortSettings() {
        const SORT_BY_KEY = 'fastpic_sort_by';
        const SORT_ORDER_KEY = 'fastpic_sort_order';
        const validSortBy = ['filename', 'folder_filename', 'modified_at', 'file_size'];
        const validSortOrder = ['asc', 'desc'];
        var sortByInput = document.getElementById('sort-by-input');
        var sortOrderInput = document.getElementById('sort-order-input');
        if (!sortByInput || !sortOrderInput) return;
        var savedBy = localStorage.getItem(SORT_BY_KEY);
        var savedOrder = localStorage.getItem(SORT_ORDER_KEY);
        if (savedBy && validSortBy.indexOf(savedBy) >= 0) sortByInput.value = savedBy;
        if (savedOrder && validSortOrder.indexOf(savedOrder) >= 0) sortOrderInput.value = savedOrder;
    })();

    function positionThumbnailSizePopover(btn, popover) {
        const rect = btn.getBoundingClientRect();
        const popW = 200;
        let top = rect.bottom + 8;
        let left = rect.left;
        if (left + popW > window.innerWidth - 8) left = window.innerWidth - popW - 8;
        if (left < 8) left = 8;
        const popH = popover.scrollHeight || 120;
        if (top + popH > window.innerHeight - 8) {
            top = rect.top - popH - 8;
            if (top < 8) top = 8;
        }
        popover.style.top = top + 'px';
        popover.style.left = left + 'px';
    }

    document.body.addEventListener('click', function(e) {
        const btn = e.target.closest('#thumbnail-size-btn');
        const popover = document.getElementById('thumbnail-size-popover');
        const insidePopover = popover && popover.contains(e.target);
        if (btn) {
            popover.classList.toggle('hidden');
            if (!popover.classList.contains('hidden')) {
                positionThumbnailSizePopover(btn, popover);
                const slider = document.getElementById('thumbnail-size-slider');
                const saved = localStorage.getItem(STORAGE_KEY);
                const cols = saved ? parseInt(saved, 10) : DEFAULT_COLS;
                if (slider) slider.value = Math.max(MIN_COLS, Math.min(MAX_COLS, cols));
                setThumbnailCols(cols, true);  // 仅同步显示，不刷新画廊（避免弹窗被替换）
            }
        } else if (popover && !popover.classList.contains('hidden') && !insidePopover) {
            popover.classList.add('hidden');
            // 关闭 popover 时再刷新画廊，使每页数量与列数匹配
            if (typeof window.refreshGallery === 'function') window.refreshGallery();
            if (typeof window.rebuildWaterfall === 'function') window.rebuildWaterfall();
        }
    });

    document.body.addEventListener('input', function(e) {
        if (e.target.id === 'thumbnail-size-slider') {
            // 拖动时用 skipRefresh=true，避免 refreshGallery 替换整个画廊导致 popover 被销毁
            setThumbnailCols(parseInt(e.target.value, 10), true);
        }
    });
})();

// 侧边栏宽度调节
(function() {
    const SIDEBAR_MIN = 160;
    const SIDEBAR_MAX = 800;
    const STORAGE_KEY = 'fastpic_sidebar_width';

    function initSidebarWidth() {
        const saved = localStorage.getItem(STORAGE_KEY);
        if (saved) {
            const w = parseInt(saved, 10);
            if (w >= SIDEBAR_MIN && w <= SIDEBAR_MAX) {
                document.documentElement.style.setProperty('--sidebar-width', w + 'px');
            }
        }
    }

    function setSidebarWidth(px) {
        const w = Math.max(SIDEBAR_MIN, Math.min(SIDEBAR_MAX, px));
        document.documentElement.style.setProperty('--sidebar-width', w + 'px');
        localStorage.setItem(STORAGE_KEY, String(w));
    }

    initSidebarWidth();

    const handle = document.getElementById('resize-handle');
    const sidebar = document.getElementById('sidebar');

    handle.addEventListener('mousedown', function(e) {
        if (e.button !== 0) return;
        e.preventDefault();
        const startX = e.clientX;
        const startWidth = sidebar.getBoundingClientRect().width;

        let rafId = null;
        let pendingWidth = null;

        function onMouseMove(e) {
            pendingWidth = startWidth + (e.clientX - startX);
            if (rafId === null) {
                rafId = requestAnimationFrame(function() {
                    setSidebarWidth(pendingWidth);
                    rafId = null;
                });
            }
        }

        function onMouseUp(e) {
            document.removeEventListener('mousemove', onMouseMove);
            document.removeEventListener('mouseup', onMouseUp);
            if (rafId !== null) {
                cancelAnimationFrame(rafId);
                rafId = null;
                setSidebarWidth(startWidth + (e.clientX - startX));
            }
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
        }

        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp);
    });
})();

// 瀑布流布局管理器（JS 多列，避免 CSS columns 重排）
(function() {
    var wfColumns = [];
    var wfHeights = [];
    var wfInitialized = false;

    function getColCount() {
        var saved = localStorage.getItem('fastpic_gallery_cols');
        return saved ? Math.max(2, Math.min(8, parseInt(saved, 10))) : 4;
    }

    /** 按视口宽度限制瀑布流最大列数，避免小屏单列过宽 */
    function getMaxWaterfallCols() {
        var w = window.innerWidth;
        if (w < 640) return 2;
        if (w < 768) return 3;
        if (w < 1024) return 4;
        if (w < 1280) return 6;
        return 8;
    }

    function isWaterfallMode() {
        var marker = document.getElementById('current-path-marker');
        return marker && marker.getAttribute('data-mode') === 'waterfall';
    }

    function getShortestCol() {
        var minIdx = 0;
        for (var i = 1; i < wfHeights.length; i++) {
            if (wfHeights[i] < wfHeights[minIdx]) minIdx = i;
        }
        return minIdx;
    }

    function getItemHeight(item) {
        var w = parseInt(item.getAttribute('data-w')) || 0;
        var h = parseInt(item.getAttribute('data-h')) || 0;
        if (w > 0 && h > 0) return (h / w) * 1000;
        return 1000;
    }

    function placeItem(item) {
        var idx = getShortestCol();
        wfColumns[idx].appendChild(item);
        wfHeights[idx] += getItemHeight(item);
    }

    /** 使用 DocumentFragment 批量 append 到各列，减少 reflow */
    function placeItemsBatch(items) {
        if (items.length === 0) return;
        var fragments = [];
        for (var i = 0; i < wfColumns.length; i++) {
            fragments.push(document.createDocumentFragment());
        }
        items.forEach(function(item) {
            var idx = getShortestCol();
            fragments[idx].appendChild(item);
            wfHeights[idx] += getItemHeight(item);
        });
        for (var j = 0; j < wfColumns.length; j++) {
            if (fragments[j].childNodes.length > 0) {
                wfColumns[j].appendChild(fragments[j]);
            }
        }
    }

    function initWaterfall() {
        var grid = document.getElementById('gallery-grid');
        if (!grid) return;

        var colCount = Math.min(getColCount(), getMaxWaterfallCols());

        // 收集所有直接子元素（图片卡片 + 滚动哨兵）
        var items = [];
        var sentinel = null;
        Array.from(grid.children).forEach(function(child) {
            if (child.classList && child.classList.contains('selectable-item')) {
                items.push(child);
            } else if (child.tagName === 'A') {
                items.push(child);
            } else if (child.hasAttribute('hx-get') && child.hasAttribute('hx-trigger')) {
                sentinel = child;
            }
        });

        // 移除旧元素
        items.forEach(function(item) { item.remove(); });
        if (sentinel) sentinel.remove();

        // 移除可能存在的旧 wf-row
        var oldRow = document.getElementById('wf-row');
        if (oldRow) oldRow.remove();

        // 创建列容器
        var row = document.createElement('div');
        row.id = 'wf-row';
        grid.appendChild(row);

        wfColumns = [];
        wfHeights = new Array(colCount).fill(0);
        for (var i = 0; i < colCount; i++) {
            var col = document.createElement('div');
            col.className = 'wf-col';
            row.appendChild(col);
            wfColumns.push(col);
        }

        // 使用 DocumentFragment 批量 append，减少 reflow
        placeItemsBatch(items);

        // 哨兵放在 grid 最后（在列容器下方，确保滚动到底部才触发）
        if (sentinel) grid.appendChild(sentinel);

        wfInitialized = true;
    }

    function distributeNew() {
        var grid = document.getElementById('gallery-grid');
        if (!grid) return;

        var wfRow = document.getElementById('wf-row');
        if (!wfRow) return;

        // 找到所有不在 wf-row 中的直接子元素（HTMX 追加的新内容）
        var newImages = [];
        var sentinels = [];
        Array.from(grid.children).forEach(function(child) {
            if (child === wfRow) return;
            if (child.classList && child.classList.contains('selectable-item')) {
                newImages.push(child);
            } else if (child.tagName === 'A') {
                newImages.push(child);
            } else if (child.hasAttribute('hx-get') && child.hasAttribute('hx-trigger')) {
                sentinels.push(child);
            }
        });

        // 从 grid 移除后，使用 DocumentFragment 批量分配到各列
        newImages.forEach(function(item) { item.remove(); });
        placeItemsBatch(newImages);

        // 只保留最新的哨兵，移除旧的
        if (sentinels.length > 1) {
            sentinels.slice(0, -1).forEach(function(s) { s.remove(); });
        }
    }

    function rebuildWaterfall() {
        if (!isWaterfallMode() || !wfInitialized) return;

        var grid = document.getElementById('gallery-grid');
        if (!grid) return;

        var wfRow = document.getElementById('wf-row');
        if (!wfRow) return;

        // 从各列中收集所有图片
        var items = [];
        wfColumns.forEach(function(col) {
            Array.from(col.children).forEach(function(child) {
                items.push(child);
            });
        });

        // 收集哨兵
        var sentinel = null;
        Array.from(grid.children).forEach(function(child) {
            if (child !== wfRow && child.hasAttribute('hx-get') && child.hasAttribute('hx-trigger')) {
                sentinel = child;
            }
        });

        // 清空
        wfRow.remove();
        if (sentinel) sentinel.remove();

        // 重新放回 grid
        items.forEach(function(item) { grid.appendChild(item); });
        if (sentinel) grid.appendChild(sentinel);

        // 重新初始化
        initWaterfall();
    }

    // 暴露给外部（缩略图大小改变时调用）
    window.rebuildWaterfall = rebuildWaterfall;

    // resize 处理已统一到 initThumbnailSize 模块：cols 更新 + syncGalleryGridCols + rebuildWaterfall

    // 监听 HTMX 交换事件
    document.body.addEventListener('htmx:afterSettle', function(ev) {
        if (ev.detail.target.id === 'gallery-container') {
            // 全量加载（首页 / 切换模式 / 切换文件夹）
            wfInitialized = false;
            wfColumns = [];
            wfHeights = [];
            if (isWaterfallMode()) {
                initWaterfall();
            }
        } else if (ev.detail.target.id === 'gallery-grid' && wfInitialized && isWaterfallMode()) {
            // 追加加载（无限滚动），延迟到下一帧避免阻塞 HTMX 交换后的渲染
            requestAnimationFrame(function() { distributeNew(); });
        } else if (ev.detail.target.id === 'gallery-grid' && !isWaterfallMode()) {
            // 文件夹模式追加加载：清理旧的滚动哨兵，只保留最新的
            var grid = document.getElementById('gallery-grid');
            if (grid) {
                var sentinels = grid.querySelectorAll(':scope > .scroll-sentinel');
                if (sentinels.length > 1) {
                    for (var i = 0; i < sentinels.length - 1; i++) {
                        sentinels[i].remove();
                    }
                }
            }
        }
        if (ev.detail.target.id === 'gallery-container' || ev.detail.target.id === 'gallery-grid') {
            if (typeof window.syncGalleryGridCols === 'function') window.syncGalleryGridCols();
        }
    });
})();

// ---------- 新建文件夹 ----------
(function() {
    function getCurrentPath() {
        var marker = document.getElementById('current-path-marker');
        return marker ? (marker.getAttribute('data-path') || '') : '';
    }

    window.showCreateFolderDialog = function() {
        var old = document.getElementById('create-folder-dialog');
        if (old) old.remove();

        var overlay = document.createElement('div');
        overlay.id = 'create-folder-dialog';
        overlay.className = 'fixed inset-0 z-[100] flex items-center justify-center bg-black/40';
        overlay.innerHTML =
            '<div class="bg-white rounded-xl shadow-2xl max-w-md w-full mx-4 p-6">' +
                '<div class="flex items-center gap-3 mb-4">' +
                    '<div class="w-10 h-10 rounded-full bg-blue-100 flex items-center justify-center flex-shrink-0">' +
                        '<svg class="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-width="2" stroke-linecap="round" stroke-linejoin="round" d="M9 13h6m-3-3v6m-9 1V7a2 2 0 012-2h6l2 2h6a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2z"/></svg>' +
                    '</div>' +
                    '<h3 class="text-lg font-semibold text-slate-800">新建文件夹</h3>' +
                '</div>' +
                '<div class="mb-2 text-sm text-slate-500">位置：<span class="font-medium text-slate-700">' + (getCurrentPath() || '根目录') + '</span></div>' +
                '<input type="text" id="new-folder-name" placeholder="输入文件夹名称" ' +
                    'class="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 mb-1">' +
                '<div id="create-folder-error" class="text-sm text-red-500 mb-4 hidden"></div>' +
                '<div class="flex justify-end gap-3 mt-4">' +
                    '<button type="button" class="cancel-btn px-4 py-2 rounded-lg border border-slate-300 text-slate-700 hover:bg-slate-50 text-sm font-medium transition-colors">取消</button>' +
                    '<button type="button" class="confirm-btn px-4 py-2 rounded-lg bg-blue-500 hover:bg-blue-600 text-white text-sm font-medium transition-colors">创建</button>' +
                '</div>' +
            '</div>';

        document.body.appendChild(overlay);

        var input = overlay.querySelector('#new-folder-name');
        var errorEl = overlay.querySelector('#create-folder-error');
        setTimeout(function() { input.focus(); }, 50);

        function doCreate() {
            var name = input.value.trim();
            if (!name) { errorEl.textContent = '名称不能为空'; errorEl.classList.remove('hidden'); return; }
            if (/[\/\\]/.test(name) || name.includes('..')) { errorEl.textContent = '名称不能包含 / \\ 或 ..'; errorEl.classList.remove('hidden'); return; }

            var confirmBtn = overlay.querySelector('.confirm-btn');
            confirmBtn.disabled = true;
            confirmBtn.textContent = '创建中...';

            fetch('/api/create-folder', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({path: getCurrentPath(), name: name})
            }).then(function(r) { return r.json(); }).then(function(data) {
                if (data.ok) {
                    overlay.remove();
                    // 刷新画廊
                    refreshGalleryFromModal();
                } else {
                    errorEl.textContent = data.error || '创建失败';
                    errorEl.classList.remove('hidden');
                    confirmBtn.disabled = false;
                    confirmBtn.textContent = '创建';
                }
            }).catch(function(err) {
                errorEl.textContent = '请求失败: ' + err;
                errorEl.classList.remove('hidden');
                confirmBtn.disabled = false;
                confirmBtn.textContent = '创建';
            });
        }

        overlay.querySelector('.cancel-btn').addEventListener('click', function() { overlay.remove(); });
        overlay.querySelector('.confirm-btn').addEventListener('click', doCreate);
        input.addEventListener('keydown', function(e) { if (e.key === 'Enter') doCreate(); if (e.key === 'Escape') overlay.remove(); });
        overlay.addEventListener('click', function(e) { if (e.target === overlay) overlay.remove(); });
    };
})();

// ---------- 重命名（分发 + 单文件夹 + 单图片 + 批量）----------
(function() {
    window.showRenameDialog = function() {
        var folders = (window.getSelectedFolders && window.getSelectedFolders()) || [];
        var images = (window.getSelectedImages && window.getSelectedImages()) || [];
        if (folders.length === 1 && images.length === 0) {
            window.showRenameFolderDialog && window.showRenameFolderDialog();
        } else if (images.length === 1 && folders.length === 0) {
            window.showRenameImageDialog && window.showRenameImageDialog();
        } else if ((folders.length > 1 && images.length === 0) || (images.length > 1 && folders.length === 0)) {
            window.showBatchRenameDialog && window.showBatchRenameDialog();
        }
    };

    window.showRenameFolderDialog = function(folderPathFromToolbar) {
        var folderPath;
        if (folderPathFromToolbar !== undefined && folderPathFromToolbar !== '') {
            folderPath = folderPathFromToolbar;
        } else {
            var folders = (window.getSelectedFolders && window.getSelectedFolders()) || [];
            if (folders.length !== 1) return;
            folderPath = folders[0];
        }
        var currentName = folderPath.split('/').pop() || folderPath;

        var old = document.getElementById('rename-folder-dialog');
        if (old) old.remove();

        var overlay = document.createElement('div');
        overlay.id = 'rename-folder-dialog';
        overlay.className = 'fixed inset-0 z-[100] flex items-center justify-center bg-black/40';
        overlay.innerHTML =
            '<div class="bg-white rounded-xl shadow-2xl max-w-md w-full mx-4 p-6">' +
                '<div class="flex items-center gap-3 mb-4">' +
                    '<div class="w-10 h-10 rounded-full bg-blue-100 flex items-center justify-center flex-shrink-0">' +
                        '<svg class="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-width="2" stroke-linecap="round" stroke-linejoin="round" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"/></svg>' +
                    '</div>' +
                    '<h3 class="text-lg font-semibold text-slate-800">重命名文件夹</h3>' +
                '</div>' +
                '<div class="mb-2 text-sm text-slate-500">路径：<span class="font-medium text-slate-700">' + (folderPath || '根目录') + '</span></div>' +
                '<input type="text" id="rename-folder-name" placeholder="输入新名称" value="' + (currentName || '').replace(/"/g, '&quot;').replace(/</g, '&lt;') + '" ' +
                    'class="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 mb-1">' +
                '<div id="rename-folder-error" class="text-sm text-red-500 mb-4 hidden"></div>' +
                '<div class="flex justify-end gap-3 mt-4">' +
                    '<button type="button" class="rename-cancel-btn px-4 py-2 rounded-lg border border-slate-300 text-slate-700 hover:bg-slate-50 text-sm font-medium transition-colors">取消</button>' +
                    '<button type="button" class="rename-confirm-btn px-4 py-2 rounded-lg bg-blue-500 hover:bg-blue-600 text-white text-sm font-medium transition-colors">重命名</button>' +
                '</div>' +
            '</div>';

        document.body.appendChild(overlay);

        var input = overlay.querySelector('#rename-folder-name');
        var errorEl = overlay.querySelector('#rename-folder-error');
        setTimeout(function() { input.focus(); input.select(); }, 50);

        function doRename() {
            var newName = input.value.trim();
            if (!newName) { errorEl.textContent = '名称不能为空'; errorEl.classList.remove('hidden'); return; }
            if (/[\/\\]/.test(newName) || newName.includes('..')) { errorEl.textContent = '名称不能包含 / \\ 或 ..'; errorEl.classList.remove('hidden'); return; }

            var confirmBtn = overlay.querySelector('.rename-confirm-btn');
            confirmBtn.disabled = true;
            confirmBtn.textContent = '重命名中...';
            if (window.showOperationLoading) window.showOperationLoading('正在重命名，请稍候...');

            fetch('/api/rename-folder', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({path: folderPath, new_name: newName})
            }).then(function(r) {
                var ct = r.headers.get('content-type') || '';
                if (ct.includes('application/json')) return r.json();
                return r.text().then(function(t) { throw new Error(r.status + ' ' + (t || r.statusText).slice(0, 80)); });
            }).then(function(data) {
                if (data && data.ok) {
                    overlay.remove();
                    if (window.exitSelectMode) window.exitSelectMode();
                    var newName = input.value.trim();
                    var parentPath = folderPath.split('/').slice(0, -1).join('/');
                    var newPath = parentPath ? parentPath + '/' + newName : newName;
                    var marker = document.getElementById('current-path-marker');
                    if (marker) marker.setAttribute('data-path', newPath);
                    var pathInput = document.querySelector('[name=path]');
                    if (pathInput) pathInput.value = newPath;
                    refreshGalleryFromModal();
                } else {
                    errorEl.textContent = (data && (data.error || data.detail)) || '重命名失败';
                    errorEl.classList.remove('hidden');
                    confirmBtn.disabled = false;
                    confirmBtn.textContent = '重命名';
                }
            }).catch(function(err) {
                errorEl.textContent = '请求失败: ' + (err && err.message ? err.message : String(err));
                errorEl.classList.remove('hidden');
                confirmBtn.disabled = false;
                confirmBtn.textContent = '重命名';
            }).finally(function() {
                if (window.hideOperationLoading) window.hideOperationLoading();
            });
        }

        overlay.querySelector('.rename-cancel-btn').addEventListener('click', function() { overlay.remove(); });
        overlay.querySelector('.rename-confirm-btn').addEventListener('click', doRename);
        input.addEventListener('keydown', function(e) { if (e.key === 'Enter') doRename(); if (e.key === 'Escape') overlay.remove(); });
        // 不响应遮罩点击关闭，避免误触（只能通过取消按钮或 Esc 关闭）
    };

    window.showRenameImageDialog = function(optImageId) {
        var imageId = optImageId;
        if (!imageId) {
            var images = (window.getSelectedImages && window.getSelectedImages()) || [];
            if (images.length !== 1) return;
            imageId = images[0];
        }
        var currentStem = '';
        var currentExt = '';
        var item = document.querySelector('.selectable-item[data-image-id="' + imageId + '"]');
        if (item) {
            var rp = item.getAttribute('data-relative-path') || '';
            var fullName = rp.split('/').pop() || '';
            var dotIdx = fullName.lastIndexOf('.');
            if (dotIdx > 0) { currentStem = fullName.slice(0, dotIdx); currentExt = fullName.slice(dotIdx); }
            else { currentStem = fullName; }
        }
        var old = document.getElementById('rename-image-dialog');
        if (old) old.remove();
        var overlay = document.createElement('div');
        overlay.id = 'rename-image-dialog';
        overlay.className = 'fixed inset-0 z-[100] flex items-center justify-center bg-black/40';
        overlay.innerHTML =
            '<div class="bg-white rounded-xl shadow-2xl max-w-md w-full mx-4 p-6">' +
                '<div class="flex items-center gap-3 mb-4">' +
                    '<div class="w-10 h-10 rounded-full bg-blue-100 flex items-center justify-center flex-shrink-0">' +
                        '<svg class="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-width="2" stroke-linecap="round" stroke-linejoin="round" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"/></svg>' +
                    '</div>' +
                    '<h3 class="text-lg font-semibold text-slate-800">重命名图片/视频</h3>' +
                '</div>' +
                '<div class="mb-2 text-sm text-slate-500">输入新名称（不含扩展名）<span id="rename-ext-hint" class="text-slate-400"></span></div>' +
                '<input type="text" id="rename-image-name" placeholder="输入新名称" value="' + (currentStem || '').replace(/"/g, '&quot;').replace(/</g, '&lt;') + '" ' +
                    'class="w-full px-3 py-2 border border-slate-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 mb-1">' +
                '<div id="rename-image-error" class="text-sm text-red-500 mb-4 hidden"></div>' +
                '<div class="flex justify-end gap-3 mt-4">' +
                    '<button type="button" class="rename-img-cancel px-4 py-2 rounded-lg border border-slate-300 text-slate-700 hover:bg-slate-50 text-sm font-medium transition-colors">取消</button>' +
                    '<button type="button" class="rename-img-confirm px-4 py-2 rounded-lg bg-blue-500 hover:bg-blue-600 text-white text-sm font-medium transition-colors">重命名</button>' +
                '</div>' +
            '</div>';
        document.body.appendChild(overlay);
        var input = overlay.querySelector('#rename-image-name');
        var errorEl = overlay.querySelector('#rename-image-error');
        var extHint = overlay.querySelector('#rename-ext-hint');
        overlay._currentExt = currentExt;
        if (extHint && currentExt) extHint.textContent = '，扩展名 ' + currentExt + ' 将保留';
        if (!currentStem && !currentExt) {
            fetch('/api/batch-rename-info', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({image_ids: [imageId], folder_paths: []})
            }).then(function(r) { return r.json(); }).then(function(data) {
                if (data && data.images && data.images[0]) {
                    var fn = data.images[0].filename || '';
                    var di = fn.lastIndexOf('.');
                    if (di > 0) { input.value = fn.slice(0, di); overlay._currentExt = fn.slice(di); }
                    else { input.value = fn; }
                    if (extHint && overlay._currentExt) extHint.textContent = '，扩展名 ' + overlay._currentExt + ' 将保留';
                }
            }).catch(function() {});
        }
        setTimeout(function() { input.focus(); input.select(); }, 50);
        function doRename() {
            var stem = input.value.trim();
            if (!stem) { errorEl.textContent = '名称不能为空'; errorEl.classList.remove('hidden'); return; }
            if (/[\/\\:*?"<>|]/.test(stem) || stem.includes('..')) { errorEl.textContent = '名称不能包含 / \\ : * ? " < > | 或 ..'; errorEl.classList.remove('hidden'); return; }
            var newFilename = stem + (overlay._currentExt || '');
            var confirmBtn = overlay.querySelector('.rename-img-confirm');
            confirmBtn.disabled = true;
            confirmBtn.textContent = '重命名中...';
            if (window.showOperationLoading) window.showOperationLoading('正在重命名，请稍候...');
            fetch('/api/rename-image', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({id: imageId, new_filename: newFilename})
            }).then(function(r) {
                var ct = r.headers.get('content-type') || '';
                if (ct.includes('application/json')) return r.json();
                return r.text().then(function(t) { throw new Error(r.status + ' ' + (t || r.statusText).slice(0, 80)); });
            }).then(function(data) {
                if (data && data.ok) {
                    overlay.remove();
                    if (window.exitSelectMode) window.exitSelectMode();
                    if (typeof refreshGalleryFromModal === 'function') refreshGalleryFromModal();
                    if (data.path && typeof modalImageIds !== 'undefined' && modalImageIds[modalIndex] === imageId) {
                        var newUrl = '/photos/' + (data.path || '').split('/').map(encodeURIComponent).join('/');
                        modalImages[modalIndex] = newUrl;
                        if (typeof _showModalContent === 'function') _showModalContent(newUrl, modalMediaTypes[modalIndex]);
                        if (typeof refreshImageInfoIfVisible === 'function') refreshImageInfoIfVisible();
                    }
                } else {
                    errorEl.textContent = (data && (data.error || data.detail)) || '重命名失败';
                    errorEl.classList.remove('hidden');
                    confirmBtn.disabled = false;
                    confirmBtn.textContent = '重命名';
                }
            }).catch(function(err) {
                errorEl.textContent = '请求失败: ' + (err && err.message ? err.message : String(err));
                errorEl.classList.remove('hidden');
                confirmBtn.disabled = false;
                confirmBtn.textContent = '重命名';
            }).finally(function() {
                if (window.hideOperationLoading) window.hideOperationLoading();
            });
        }
        overlay.querySelector('.rename-img-cancel').addEventListener('click', function() { overlay.remove(); });
        overlay.querySelector('.rename-img-confirm').addEventListener('click', doRename);
        input.addEventListener('keydown', function(e) { if (e.key === 'Enter') doRename(); if (e.key === 'Escape') overlay.remove(); });
    };

    window.showBatchRenameDialog = function() {
        var folders = (window.getSelectedFolders && window.getSelectedFolders()) || [];
        var images = (window.getSelectedImages && window.getSelectedImages()) || [];
        if ((folders.length <= 1 && images.length === 0) || (images.length <= 1 && folders.length === 0)) return;
        var old = document.getElementById('batch-rename-dialog');
        if (old) old.remove();
        var overlay = document.createElement('div');
        overlay.id = 'batch-rename-dialog';
        overlay.className = 'fixed inset-0 z-[100] flex items-center justify-center bg-black/40 p-4';
        overlay.innerHTML =
            '<div class="bg-white rounded-xl shadow-2xl max-w-2xl w-full max-h-[90vh] flex flex-col" onclick="event.stopPropagation()">' +
                '<div class="flex items-center gap-3 p-4 border-b border-slate-200 flex-shrink-0">' +
                    '<div class="w-10 h-10 rounded-full bg-blue-100 flex items-center justify-center flex-shrink-0">' +
                        '<svg class="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-width="2" stroke-linecap="round" stroke-linejoin="round" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"/></svg>' +
                    '</div>' +
                    '<h3 class="text-lg font-semibold text-slate-800">批量重命名</h3>' +
                '</div>' +
                '<div class="p-4 border-b border-slate-200 flex-shrink-0">' +
                    '<div class="flex items-center justify-between mb-2">' +
                        '<span class="text-sm font-medium text-slate-700">规则（按顺序执行）</span>' +
                        '<button type="button" id="batch-add-rule" class="text-sm px-3 py-1.5 rounded-lg bg-blue-100 text-blue-700 hover:bg-blue-200 transition-colors">+ 添加规则</button>' +
                    '</div>' +
                    '<div id="batch-rules-list" class="space-y-2 max-h-40 overflow-y-auto"></div>' +
                '</div>' +
                '<div class="flex-1 min-h-0 overflow-auto p-4">' +
                    '<div class="flex items-center justify-between gap-2 mb-2">' +
                        '<span class="text-sm text-slate-500">预览</span>' +
                        '<div class="flex items-center gap-2">' +
                            '<label class="text-xs text-slate-500">排序：</label>' +
                            '<select id="batch-preview-sort" class="text-sm px-2 py-1 border border-slate-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500">' +
                                '<option value="original">原序</option>' +
                                '<option value="name_asc">文件名升序</option>' +
                                '<option value="name_desc">文件名降序</option>' +
                                '<option value="mtime_asc">修改时间升序</option>' +
                                '<option value="mtime_desc">修改时间降序</option>' +
                            '</select>' +
                        '</div>' +
                    '</div>' +
                    '<div id="batch-rename-preview" class="border border-slate-200 rounded-lg overflow-hidden text-sm max-h-48 overflow-y-auto"></div>' +
                '</div>' +
                '<div class="p-4 border-t border-slate-200 flex justify-end gap-3 flex-shrink-0">' +
                    '<button type="button" class="batch-rename-cancel px-4 py-2 rounded-lg border border-slate-300 text-slate-700 hover:bg-slate-50 text-sm font-medium transition-colors">取消</button>' +
                    '<button type="button" class="batch-rename-confirm px-4 py-2 rounded-lg bg-blue-500 hover:bg-blue-600 text-white text-sm font-medium transition-colors">执行重命名</button>' +
                '</div>' +
            '</div>';
        document.body.appendChild(overlay);
        overlay._rules = [{ type: 'replace', find: '', replace: '', regex: false }];
        renderRulesList();

        var items = [];
        var isFolder = folders.length > 0;
        fetch('/api/batch-rename-info', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({image_ids: images, folder_paths: folders})
        }).then(function(r) { return r.json(); }).then(function(data) {
            if (data.folders) items = items.concat(data.folders.map(function(f) { return {type: 'folder', path: f.path, name: f.name, modified_at: f.modified_at || 0}; }));
            if (data.images) items = items.concat(data.images.map(function(i) { return {type: 'image', id: i.id, name: i.filename, path: i.relative_path, modified_at: i.modified_at || 0}; }));
            renderBatchPreview();
        }).catch(function() {
            overlay.querySelector('#batch-rename-preview').innerHTML = '<div class="p-4 text-red-500">加载失败</div>';
        });

        function getStemAndExt(filename) {
            if (!filename) return {stem: '', ext: ''};
            if (filename.charAt(0) === '.') return {stem: '', ext: filename};
            var i = filename.lastIndexOf('.');
            if (i <= 0) return {stem: filename, ext: ''};
            return {stem: filename.slice(0, i), ext: filename.slice(i)};
        }
        function applySingleRule(name, isMedia, rule, indexForSequence) {
            var result = name;
            if (rule.type === 'sequence' && indexForSequence !== undefined) {
                var start = parseInt(rule.seqStart, 10) || 0;
                var pad = Math.max(0, Math.min(10, parseInt(rule.seqPad, 10) || 0));
                var sep = (rule.seqSep || '').replace(/[\/\\:*?"<>|]/g, '') || '_';
                var num = start + indexForSequence;
                var seqStr = pad > 0 ? String(num).padStart(pad, '0') : String(num);
                var asPrefix = rule.seqPos !== 'suffix';
                if (isMedia) {
                    var se = getStemAndExt(name);
                    result = asPrefix ? seqStr + sep + se.stem + se.ext : se.stem + sep + seqStr + se.ext;
                } else {
                    result = asPrefix ? seqStr + sep + name : name + sep + seqStr;
                }
            } else if (rule.type === 'replace') {
                var find = rule.find || '';
                var repl = rule.replace || '';
                if (!find) return name;
                try {
                    if (isMedia) {
                        var se = getStemAndExt(name);
                        var newStem = rule.regex ? se.stem.replace(new RegExp(find, 'g'), repl) : se.stem.split(find).join(repl);
                        result = newStem + se.ext;
                    } else {
                        result = rule.regex ? name.replace(new RegExp(find, 'g'), repl) : name.split(find).join(repl);
                    }
                } catch (e) { return {error: e.message}; }
            } else if (rule.type === 'prefix') {
                var prefix = rule.prefix || '';
                if (isMedia) { var se = getStemAndExt(name); result = prefix + se.stem + se.ext; }
                else result = prefix + name;
            } else if (rule.type === 'suffix') {
                var suffix = rule.suffix || '';
                if (isMedia) { var se = getStemAndExt(name); result = se.stem + suffix + se.ext; }
                else result = name + suffix;
            } else if (rule.type === 'delete') {
                var n = parseInt(rule.deleteN, 10) || 0;
                var fromEnd = rule.deletePos === 'end';
                if (isMedia) { var se = getStemAndExt(name); var s = se.stem; result = fromEnd ? s.slice(0, Math.max(0, s.length - n)) + se.ext : s.slice(n) + se.ext; }
                else result = fromEnd ? name.slice(0, Math.max(0, name.length - n)) : name.slice(n);
            }
            return result;
        }
        function applyAllRules(name, isMedia, indexForSequence) {
            var r = name;
            for (var i = 0; i < overlay._rules.length; i++) {
                r = applySingleRule(r, isMedia, overlay._rules[i], indexForSequence);
                if (r && typeof r === 'object' && r.error) return r;
            }
            return r;
        }
        function renderRuleCard(idx, rule) {
            var type = rule.type || 'replace';
            var card = document.createElement('div');
            card.className = 'batch-rule-card flex flex-wrap items-start gap-2 p-2 rounded-lg border border-slate-200 bg-slate-50';
            card.dataset.ruleIdx = idx;
            var typeSelect = '<select class="rule-type-select text-sm px-2 py-1.5 border border-slate-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500">' +
                '<option value="replace"' + (type === 'replace' ? ' selected' : '') + '>查找替换</option>' +
                '<option value="prefix"' + (type === 'prefix' ? ' selected' : '') + '>添加前缀</option>' +
                '<option value="suffix"' + (type === 'suffix' ? ' selected' : '') + '>添加后缀</option>' +
                '<option value="delete"' + (type === 'delete' ? ' selected' : '') + '>删除字符</option>' +
                '<option value="sequence"' + (type === 'sequence' ? ' selected' : '') + '>添加序号</option>' +
            '</select>';
            var configHtml = '';
            if (type === 'replace') {
                configHtml = '<div class="flex-1 min-w-0 flex flex-wrap gap-2 items-center">' +
                    '<input type="text" class="rule-find px-2 py-1.5 border border-slate-300 rounded text-sm flex-1 min-w-[80px]" placeholder="查找" value="' + escapeAttr(rule.find || '') + '">' +
                    '<input type="text" class="rule-replace px-2 py-1.5 border border-slate-300 rounded text-sm flex-1 min-w-[80px]" placeholder="替换为" value="' + escapeAttr(rule.replace || '') + '">' +
                    '<label class="flex items-center gap-1 cursor-pointer text-xs"><input type="checkbox" class="rule-regex accent-blue-500"' + (rule.regex ? ' checked' : '') + '> 正则</label>' +
                '</div>';
            } else if (type === 'prefix') {
                configHtml = '<input type="text" class="rule-prefix flex-1 min-w-[120px] px-2 py-1.5 border border-slate-300 rounded text-sm" placeholder="前缀" value="' + escapeAttr(rule.prefix || '') + '">';
            } else if (type === 'suffix') {
                configHtml = '<input type="text" class="rule-suffix flex-1 min-w-[120px] px-2 py-1.5 border border-slate-300 rounded text-sm" placeholder="后缀" value="' + escapeAttr(rule.suffix || '') + '">';
            } else if (type === 'delete') {
                configHtml = '<div class="flex flex-wrap gap-2 items-center">' +
                    '<input type="number" class="rule-delete-n w-14 px-2 py-1.5 border border-slate-300 rounded text-sm" min="0" value="' + (rule.deleteN || 1) + '">' +
                    '<span class="text-xs">个字符</span>' +
                    '<label class="flex items-center gap-1 cursor-pointer text-xs"><input type="radio" name="rule-del-pos-' + idx + '" value="start" class="accent-blue-500"' + (rule.deletePos !== 'end' ? ' checked' : '') + '> 从头</label>' +
                    '<label class="flex items-center gap-1 cursor-pointer text-xs"><input type="radio" name="rule-del-pos-' + idx + '" value="end" class="accent-blue-500"' + (rule.deletePos === 'end' ? ' checked' : '') + '> 从尾</label>' +
                '</div>';
            } else if (type === 'sequence') {
                configHtml = '<div class="flex flex-wrap gap-2 items-center">' +
                    '<input type="number" class="rule-seq-start w-14 px-2 py-1.5 border border-slate-300 rounded text-sm" min="0" value="' + (rule.seqStart || 0) + '" placeholder="起始">' +
                    '<input type="number" class="rule-seq-pad w-14 px-2 py-1.5 border border-slate-300 rounded text-sm" min="0" value="' + (rule.seqPad || 2) + '" placeholder="补零">' +
                    '<input type="text" class="rule-seq-sep w-10 px-2 py-1.5 border border-slate-300 rounded text-sm" value="' + escapeAttr(rule.seqSep || '_') + '" maxlength="2" placeholder="_">' +
                    '<label class="flex items-center gap-1 cursor-pointer text-xs"><input type="radio" name="rule-seq-pos-' + idx + '" value="prefix" class="accent-blue-500"' + (rule.seqPos !== 'suffix' ? ' checked' : '') + '> 前缀</label>' +
                    '<label class="flex items-center gap-1 cursor-pointer text-xs"><input type="radio" name="rule-seq-pos-' + idx + '" value="suffix" class="accent-blue-500"' + (rule.seqPos === 'suffix' ? ' checked' : '') + '> 后缀</label>' +
                    '<span class="text-xs text-slate-500">按预览区排序</span>' +
                '</div>';
            }
            card.innerHTML = '<div class="flex items-center gap-1 text-slate-500">' +
                '<span class="rule-order text-xs font-medium">' + (idx + 1) + '</span>' +
                '</div>' + typeSelect + '<div class="rule-config flex-1 min-w-0 flex flex-wrap gap-2">' + configHtml + '</div>' +
                '<div class="flex gap-1">' +
                '<button type="button" class="rule-move-up px-2 py-1 text-slate-500 hover:text-slate-700 hover:bg-slate-200 rounded text-xs" title="上移">↑</button>' +
                '<button type="button" class="rule-move-down px-2 py-1 text-slate-500 hover:text-slate-700 hover:bg-slate-200 rounded text-xs" title="下移">↓</button>' +
                '<button type="button" class="rule-delete px-2 py-1 text-red-500 hover:text-red-700 hover:bg-red-50 rounded text-xs">删除</button>' +
                '</div>';
            return card;
        }
        function syncRuleFromCard(card) {
            var idx = parseInt(card.dataset.ruleIdx, 10);
            var rule = overlay._rules[idx];
            if (!rule) return;
            rule.type = card.querySelector('.rule-type-select').value;
            var findIn = card.querySelector('.rule-find');
            if (findIn) rule.find = findIn.value;
            var replaceIn = card.querySelector('.rule-replace');
            if (replaceIn) rule.replace = replaceIn.value;
            var regexIn = card.querySelector('.rule-regex');
            if (regexIn) rule.regex = regexIn.checked;
            var prefixIn = card.querySelector('.rule-prefix');
            if (prefixIn) rule.prefix = prefixIn.value;
            var suffixIn = card.querySelector('.rule-suffix');
            if (suffixIn) rule.suffix = suffixIn.value;
            var delNIn = card.querySelector('.rule-delete-n');
            if (delNIn) rule.deleteN = delNIn.value;
            var delPosIn = card.querySelector('input[name="rule-del-pos-' + idx + '"]:checked');
            if (delPosIn) rule.deletePos = delPosIn.value;
            var seqStartIn = card.querySelector('.rule-seq-start');
            if (seqStartIn) rule.seqStart = seqStartIn.value;
            var seqPadIn = card.querySelector('.rule-seq-pad');
            if (seqPadIn) rule.seqPad = seqPadIn.value;
            var seqSepIn = card.querySelector('.rule-seq-sep');
            if (seqSepIn) rule.seqSep = seqSepIn.value;
            var seqPosIn = card.querySelector('input[name="rule-seq-pos-' + idx + '"]:checked');
            if (seqPosIn) rule.seqPos = seqPosIn.value;
        }
        function renderRulesList() {
            var list = overlay.querySelector('#batch-rules-list');
            if (!list) return;
            list.innerHTML = '';
            overlay._rules.forEach(function(rule, idx) {
                var card = renderRuleCard(idx, rule);
                list.appendChild(card);
                card.querySelector('.rule-type-select').addEventListener('change', function() {
                    var defaults = { replace: { find: '', replace: '', regex: false }, prefix: { prefix: '' }, suffix: { suffix: '' }, delete: { deleteN: '1', deletePos: 'start' }, sequence: { seqStart: '0', seqPad: '2', seqSep: '_', seqPos: 'prefix' } };
                    overlay._rules[idx] = { type: this.value };
                    var d = defaults[this.value] || {};
                    for (var k in d) overlay._rules[idx][k] = d[k];
                    renderRulesList();
                    renderBatchPreview();
                });
                card.querySelectorAll('.rule-find, .rule-replace, .rule-prefix, .rule-suffix, .rule-delete-n, .rule-seq-start, .rule-seq-pad, .rule-seq-sep').forEach(function(inp) {
                    inp.addEventListener('input', function() { syncRuleFromCard(card); renderBatchPreview(); });
                });
                card.querySelectorAll('.rule-regex, input[name^="rule-del-pos-"], input[name^="rule-seq-pos-"]').forEach(function(inp) {
                    inp.addEventListener('change', function() { syncRuleFromCard(card); renderBatchPreview(); });
                });
                var delBtn = card.querySelector('.rule-delete');
                if (delBtn) delBtn.addEventListener('click', function() {
                    overlay._rules.splice(idx, 1);
                    if (overlay._rules.length === 0) overlay._rules.push({ type: 'replace', find: '', replace: '', regex: false });
                    renderRulesList();
                    renderBatchPreview();
                });
                var upBtn = card.querySelector('.rule-move-up');
                if (upBtn) upBtn.addEventListener('click', function() {
                    if (idx > 0) { var t = overlay._rules[idx]; overlay._rules[idx] = overlay._rules[idx - 1]; overlay._rules[idx - 1] = t; renderRulesList(); renderBatchPreview(); }
                });
                var downBtn = card.querySelector('.rule-move-down');
                if (downBtn) downBtn.addEventListener('click', function() {
                    if (idx < overlay._rules.length - 1) { var t = overlay._rules[idx]; overlay._rules[idx] = overlay._rules[idx + 1]; overlay._rules[idx + 1] = t; renderRulesList(); renderBatchPreview(); }
                });
            });
        }
        function getSortedItems() {
            var sortVal = (overlay.querySelector('#batch-preview-sort') && overlay.querySelector('#batch-preview-sort').value) || 'original';
            if (sortVal === 'original') return items;
            var copy = items.slice();
            if (sortVal === 'name_asc' || sortVal === 'name_desc') {
                var nameCmp = function(a, b) { return (a.name || '').localeCompare(b.name || '', undefined, { numeric: true }); };
                copy.sort(sortVal === 'name_desc' ? function(a, b) { return -nameCmp(a, b); } : nameCmp);
            } else if (sortVal === 'mtime_asc' || sortVal === 'mtime_desc') {
                var mtimeCmp = function(a, b) {
                    var ma = a.modified_at || 0, mb = b.modified_at || 0;
                    if (ma !== mb) return ma - mb;
                    return (a.name || '').localeCompare(b.name || '', undefined, { numeric: true });
                };
                copy.sort(sortVal === 'mtime_desc' ? function(a, b) { return -mtimeCmp(a, b); } : mtimeCmp);
            }
            return copy;
        }
        function renderBatchPreview() {
            var sortedItems = getSortedItems();
            var hasSeq = overlay._rules.some(function(r) { return r.type === 'sequence'; });
            var tbody = [];
            var hasError = false;
            var folderRenames = [];
            var imageRenames = [];
            for (var i = 0; i < sortedItems.length; i++) {
                var it = sortedItems[i];
                var isMedia = it.type === 'image';
                var newName = applyAllRules(it.name, isMedia, hasSeq ? i : undefined);
                if (newName && typeof newName === 'object' && newName.error) { tbody.push('<tr class="bg-red-50"><td class="px-3 py-2">' + escapeHtml(it.name) + '</td><td class="px-3 py-2 text-red-600">正则错误: ' + escapeHtml(newName.error) + '</td></tr>'); hasError = true; continue; }
                if (!newName || /[\/\\:*?"<>|]/.test(newName) || newName.includes('..')) { tbody.push('<tr class="bg-red-50"><td class="px-3 py-2">' + escapeHtml(it.name) + '</td><td class="px-3 py-2 text-red-600">无效名称</td></tr>'); hasError = true; continue; }
                if (isMedia) imageRenames.push({id: it.id, new_filename: newName});
                else folderRenames.push({path: it.path, new_name: newName});
                var changed = it.name !== newName;
                tbody.push('<tr class="' + (changed ? '' : 'bg-slate-50') + '"><td class="px-3 py-2 text-slate-600">' + escapeHtml(it.name) + '</td><td class="px-3 py-2 ' + (changed ? 'text-slate-800 font-medium' : 'text-slate-400') + '">' + escapeHtml(newName) + (changed ? '' : ' (无变化)') + '</td></tr>');
            }
            overlay.querySelector('#batch-rename-preview').innerHTML = '<table class="w-full"><thead><tr class="bg-slate-100"><th class="px-3 py-2 text-left text-slate-600">原名称</th><th class="px-3 py-2 text-left text-slate-600">新名称</th></tr></thead><tbody>' + tbody.join('') + '</tbody></table>';
            overlay._folderRenames = folderRenames;
            overlay._imageRenames = imageRenames;
            overlay._hasError = hasError;
        }
        overlay.querySelector('#batch-add-rule').addEventListener('click', function() {
            overlay._rules.push({ type: 'replace', find: '', replace: '', regex: false });
            renderRulesList();
            renderBatchPreview();
        });
        var sortSelect = overlay.querySelector('#batch-preview-sort');
        if (sortSelect) sortSelect.addEventListener('change', renderBatchPreview);

        overlay.querySelector('.batch-rename-cancel').addEventListener('click', function() { overlay.remove(); });
        overlay.querySelector('.batch-rename-confirm').addEventListener('click', function() {
            if (!overlay._folderRenames && !overlay._imageRenames) return;
            if (overlay._hasError) return;
            var anyChange = overlay._folderRenames.some(function(r) { return true; }) || overlay._imageRenames.some(function(r) { return true; });
            var confirmBtn = overlay.querySelector('.batch-rename-confirm');
            confirmBtn.disabled = true;
            confirmBtn.textContent = '执行中...';
            if (window.showOperationLoading) window.showOperationLoading('正在批量重命名，请稍候...');
            fetch('/api/batch-rename', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({folder_renames: overlay._folderRenames || [], image_renames: overlay._imageRenames || []})
            }).then(function(r) {
                var ct = r.headers.get('content-type') || '';
                if (ct.includes('application/json')) return r.json();
                return r.text().then(function(t) { throw new Error(r.status + ' ' + (t || r.statusText).slice(0, 80)); });
            }).then(function(data) {
                overlay.remove();
                if (window.exitSelectMode) window.exitSelectMode();
                if (typeof refreshGalleryFromModal === 'function') refreshGalleryFromModal();
                if (data && data.errors && data.errors.length) { alert('部分失败: ' + data.errors.join('; ')); }
            }).catch(function(err) {
                confirmBtn.disabled = false;
                confirmBtn.textContent = '执行重命名';
                alert('请求失败: ' + (err && err.message ? err.message : String(err)));
            }).finally(function() {
                if (window.hideOperationLoading) window.hideOperationLoading();
            });
        });
        // 不响应遮罩点击关闭，防止误操作（只能通过取消按钮或 Esc 关闭）
    };
})();

// ---------- 上传图片/视频 ----------
(function() {
    function getCurrentPath() {
        var marker = document.getElementById('current-path-marker');
        return marker ? (marker.getAttribute('data-path') || '') : '';
    }

    window.showUploadDialog = function() {
        var old = document.getElementById('upload-dialog');
        if (old) old.remove();

        var overlay = document.createElement('div');
        overlay.id = 'upload-dialog';
        overlay.className = 'fixed inset-0 z-[100] flex items-center justify-center bg-black/40';
        overlay.innerHTML =
            '<div class="bg-white rounded-xl shadow-2xl max-w-lg w-full mx-4 p-6">' +
                '<div class="flex items-center gap-3 mb-4">' +
                    '<div class="w-10 h-10 rounded-full bg-blue-100 flex items-center justify-center flex-shrink-0">' +
                        '<svg class="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-width="2" stroke-linecap="round" stroke-linejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"/></svg>' +
                    '</div>' +
                    '<h3 class="text-lg font-semibold text-slate-800">上传图片/视频</h3>' +
                '</div>' +
                '<div class="mb-3 text-sm text-slate-500">上传到：<span class="font-medium text-slate-700">' + (getCurrentPath() || '根目录') + '</span></div>' +
                '<div id="upload-drop-zone" class="border-2 border-dashed border-slate-300 rounded-lg p-8 text-center hover:border-blue-400 transition-colors cursor-pointer">' +
                    '<svg class="w-10 h-10 mx-auto mb-3 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909M3.75 21h16.5a2.25 2.25 0 002.25-2.25V5.25a2.25 2.25 0 00-2.25-2.25H3.75a2.25 2.25 0 00-2.25 2.25v13.5a2.25 2.25 0 002.25 2.25z"/></svg>' +
                    '<p class="text-sm text-slate-600 mb-1">拖拽图片、视频或文件夹到此处，或</p>' +
                    '<p class="text-sm">' +
                    '<button type="button" class="browse-file-btn text-blue-600 hover:text-blue-700 font-medium underline">浏览文件</button>' +
                    '<span class="browse-folder-wrap"><span class="text-slate-400 mx-1">/</span><button type="button" class="browse-folder-btn text-blue-600 hover:text-blue-700 font-medium underline">浏览文件夹</button></span>' +
                    '</p>' +
                    '<input type="file" id="upload-file-input" multiple accept="image/*,video/*,.ts,video/mp2t" class="hidden">' +
                    '<input type="file" id="upload-folder-input" webkitdirectory directory class="hidden">' +
                    '<p class="text-xs text-slate-400 mt-2">支持 JPG、PNG、GIF、WebP、AVIF、BMP、MP4、WebM、MOV、MKV、TS</p>' +
                '</div>' +
                '<div class="mt-3 flex items-center gap-3 text-sm text-slate-600">' +
                    '<span class="flex-shrink-0">重复文件：</span>' +
                    '<label class="flex items-center gap-1 cursor-pointer"><input type="radio" name="on_duplicate" value="skip" checked class="accent-blue-500"> 跳过</label>' +
                    '<label class="flex items-center gap-1 cursor-pointer"><input type="radio" name="on_duplicate" value="rename" class="accent-blue-500"> 重命名</label>' +
                    '<label class="flex items-center gap-1 cursor-pointer"><input type="radio" name="on_duplicate" value="overwrite" class="accent-blue-500"> 覆盖</label>' +
                '</div>' +
                '<div id="upload-file-list" class="mt-3 max-h-40 overflow-y-auto hidden"></div>' +
                '<div id="upload-progress-wrapper" class="mt-3 hidden">' +
                    '<div class="flex items-center justify-between text-sm text-slate-600 mb-1">' +
                        '<span id="upload-progress-text">上传中...</span>' +
                        '<span id="upload-progress-count"></span>' +
                    '</div>' +
                    '<div class="w-full bg-slate-200 rounded-full h-2">' +
                        '<div id="upload-progress-bar" class="bg-blue-500 h-2 rounded-full transition-all" style="width: 0%"></div>' +
                    '</div>' +
                '</div>' +
                '<div id="upload-error" class="text-sm text-red-500 mt-2 hidden"></div>' +
                '<div class="flex justify-end gap-3 mt-4">' +
                    '<button type="button" class="cancel-btn px-4 py-2 rounded-lg border border-slate-300 text-slate-700 hover:bg-slate-50 text-sm font-medium transition-colors">取消</button>' +
                    '<button type="button" class="upload-confirm-btn px-4 py-2 rounded-lg bg-blue-500 hover:bg-blue-600 text-white text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed" disabled>上传</button>' +
                '</div>' +
            '</div>';

        document.body.appendChild(overlay);

        var dropZone = overlay.querySelector('#upload-drop-zone');
        var fileInput = overlay.querySelector('#upload-file-input');
        var folderInput = overlay.querySelector('#upload-folder-input');
        var browseFileBtn = overlay.querySelector('.browse-file-btn');
        var browseFolderBtn = overlay.querySelector('.browse-folder-btn');
        var browseFolderWrap = overlay.querySelector('.browse-folder-wrap');
        if (!('webkitdirectory' in document.createElement('input')) && browseFolderWrap) {
            browseFolderWrap.style.display = 'none';
        }
        var fileList = overlay.querySelector('#upload-file-list');
        var confirmBtn = overlay.querySelector('.upload-confirm-btn');
        var progressWrapper = overlay.querySelector('#upload-progress-wrapper');
        var progressBar = overlay.querySelector('#upload-progress-bar');
        var progressText = overlay.querySelector('#upload-progress-text');
        var progressCount = overlay.querySelector('#upload-progress-count');
        var errorEl = overlay.querySelector('#upload-error');
        var selectedFiles = [];

        function updateFileList() {
            if (selectedFiles.length === 0) {
                fileList.classList.add('hidden');
                confirmBtn.disabled = true;
                return;
            }
            fileList.classList.remove('hidden');
            confirmBtn.disabled = false;
            var html = '';
            for (var i = 0; i < selectedFiles.length; i++) {
                var item = selectedFiles[i];
                var f = item.file;
                var displayName = item.relativePath || f.name;
                var sizeMB = (f.size / 1024 / 1024).toFixed(1);
                html += '<div class="flex items-center justify-between py-1.5 px-2 text-sm ' + (i > 0 ? 'border-t border-slate-100' : '') + '">' +
                    '<span class="text-slate-700 truncate flex-1 mr-2" title="' + (displayName.replace(/"/g, '&quot;')) + '">' + displayName + '</span>' +
                    '<span class="text-slate-400 text-xs flex-shrink-0">' + sizeMB + ' MB</span>' +
                    '<button type="button" class="remove-file ml-2 text-slate-400 hover:text-red-500 flex-shrink-0" data-idx="' + i + '">' +
                        '<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-width="2" stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"/></svg>' +
                    '</button>' +
                '</div>';
            }
            fileList.innerHTML = html;
            confirmBtn.textContent = '上传 (' + selectedFiles.length + ')';

            fileList.querySelectorAll('.remove-file').forEach(function(btn) {
                btn.addEventListener('click', function() {
                    selectedFiles.splice(parseInt(this.getAttribute('data-idx')), 1);
                    updateFileList();
                });
            });
        }

        var mediaExts = /\.(jpg|jpeg|png|gif|webp|avif|bmp|mp4|webm|mov|mkv|ts)(\?|$)/i;
        function isMediaFile(file) {
            var t = (file.type || '').toLowerCase();
            var name = (file.name || '').toLowerCase();
            return t.startsWith('image/') || t.startsWith('video/') || mediaExts.test(name);
        }

        function addFiles(files) {
            for (var i = 0; i < files.length; i++) {
                var f = files[i];
                if (!isMediaFile(f)) continue;
                var relPath = (f.webkitRelativePath || '').trim();
                selectedFiles.push({ file: f, relativePath: relPath });
            }
            updateFileList();
        }

        function addFilesFromEntries(items, callback) {
            var pending = 0;
            function maybeDone() {
                if (pending === 0 && callback) callback();
            }
            function processFileEntry(entry, pathPrefix) {
                pending++;
                entry.file(function(file) {
                    if (isMediaFile(file)) {
                        var relPath = (pathPrefix ? pathPrefix + '/' : '') + entry.name;
                        selectedFiles.push({ file: file, relativePath: relPath });
                    }
                    pending--;
                    maybeDone();
                }, function() { pending--; maybeDone(); });
            }
            function processDirEntry(dirEntry, pathPrefix) {
                var dirPath = (pathPrefix ? pathPrefix + '/' : '') + dirEntry.name;
                var reader = dirEntry.createReader();
                function readBatch() {
                    reader.readEntries(function(entries) {
                        if (entries.length === 0) {
                            pending--;
                            maybeDone();
                            return;
                        }
                        for (var j = 0; j < entries.length; j++) {
                            if (entries[j].isFile) {
                                processFileEntry(entries[j], dirPath);
                            } else if (entries[j].isDirectory) {
                                processDirEntry(entries[j], dirPath);
                            }
                        }
                        readBatch();
                    }, function() { pending--; maybeDone(); });
                }
                pending++;
                readBatch();
            }
            for (var i = 0; i < items.length; i++) {
                var getEntry = items[i].webkitGetAsEntry || items[i].getAsEntry;
                if (!getEntry) continue;
                var entry = getEntry.call(items[i]);
                if (!entry) continue;
                if (entry.isFile) {
                    processFileEntry(entry, '');
                } else if (entry.isDirectory) {
                    processDirEntry(entry, '');
                }
            }
            maybeDone();
        }

        browseFileBtn.addEventListener('click', function(e) { e.stopPropagation(); fileInput.click(); });
        if (browseFolderBtn) browseFolderBtn.addEventListener('click', function(e) { e.stopPropagation(); folderInput.click(); });
        dropZone.addEventListener('click', function(e) {
            if (e.target.closest('.browse-folder-btn')) return;
            fileInput.click();
        });
        fileInput.addEventListener('change', function() {
            addFiles(this.files);
            this.value = '';
        });
        folderInput.addEventListener('change', function() {
            addFiles(this.files);
            this.value = '';
        });

        dropZone.addEventListener('dragover', function(e) { e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; this.classList.add('border-blue-400', 'bg-blue-50'); });
        dropZone.addEventListener('dragleave', function(e) { e.preventDefault(); this.classList.remove('border-blue-400', 'bg-blue-50'); });
        dropZone.addEventListener('drop', function(e) {
            e.preventDefault();
            this.classList.remove('border-blue-400', 'bg-blue-50');
            var items = e.dataTransfer.items;
            if (items && items.length > 0) {
                var hasEntry = false;
                for (var i = 0; i < items.length; i++) {
                    var getEntry = items[i].webkitGetAsEntry || items[i].getAsEntry;
                    if (getEntry && getEntry.call(items[i])) { hasEntry = true; break; }
                }
                if (hasEntry) {
                    addFilesFromEntries(items, function() { updateFileList(); });
                    return;
                }
            }
            if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files);
        });

        confirmBtn.addEventListener('click', function() {
            if (selectedFiles.length === 0) return;
            confirmBtn.disabled = true;
            progressWrapper.classList.remove('hidden');
            dropZone.classList.add('hidden');
            fileList.classList.add('hidden');

            var formData = new FormData();
            formData.append('path', getCurrentPath());
            var dupRadio = overlay.querySelector('input[name="on_duplicate"]:checked');
            formData.append('on_duplicate', dupRadio ? dupRadio.value : 'skip');
            var filePaths = [];
            for (var i = 0; i < selectedFiles.length; i++) {
                var item = selectedFiles[i];
                filePaths.push(item.relativePath || item.file.name);
                formData.append('files', item.file);
            }
            formData.append('file_paths', JSON.stringify(filePaths));

            var xhr = new XMLHttpRequest();
            xhr.open('POST', '/api/upload');
            xhr.timeout = 600000;

            xhr.upload.addEventListener('progress', function(e) {
                if (e.lengthComputable) {
                    var pct = Math.round((e.loaded / e.total) * 100);
                    progressBar.style.width = pct + '%';
                    progressCount.textContent = pct + '%';
                    if (pct >= 100) {
                        progressText.textContent = '处理中...';
                        progressBar.classList.add('upload-progress-indeterminate');
                    }
                }
            });

            xhr.addEventListener('load', function() {
                progressBar.classList.remove('upload-progress-indeterminate');
                if (xhr.status === 200) {
                    var data;
                    try {
                        data = JSON.parse(xhr.responseText);
                    } catch (err) {
                        progressText.textContent = '上传失败';
                        errorEl.textContent = '服务器返回格式错误';
                        errorEl.classList.remove('hidden');
                        confirmBtn.disabled = false;
                        return;
                    }
                    progressText.textContent = '上传完成';
                    progressBar.style.width = '100%';
                    var summary = data.uploaded + ' 个成功';
                    if (data.skipped > 0) summary += '，' + data.skipped + ' 个跳过（重复）';
                    progressCount.textContent = summary;
                    if (data.errors && data.errors.length > 0) {
                        errorEl.textContent = '部分失败: ' + data.errors.join('; ');
                        errorEl.classList.remove('hidden');
                    }
                    var delay = (data.errors && data.errors.length > 0) ? 2500 : 500;
                    setTimeout(function() {
                        overlay.remove();
                        refreshGalleryFromModal();
                    }, delay);
                } else {
                    progressText.textContent = '上传失败';
                    progressBar.style.width = '100%';
                    var errMsg = '服务器返回 ' + xhr.status;
                    if (xhr.status === 413) errMsg = '文件过大，请减少数量或分批上传';
                    else if (xhr.responseText) errMsg += ': ' + String(xhr.responseText).substring(0, 100);
                    errorEl.textContent = errMsg;
                    errorEl.classList.remove('hidden');
                    confirmBtn.disabled = false;
                }
            });

            xhr.addEventListener('error', function() {
                progressBar.classList.remove('upload-progress-indeterminate');
                progressText.textContent = '上传失败';
                progressBar.style.width = '100%';
                errorEl.textContent = '网络错误';
                errorEl.classList.remove('hidden');
                confirmBtn.disabled = false;
            });

            xhr.addEventListener('timeout', function() {
                progressBar.classList.remove('upload-progress-indeterminate');
                progressText.textContent = '上传超时';
                progressBar.style.width = '100%';
                errorEl.textContent = '服务器处理时间过长，请尝试减少文件数量或分批上传';
                errorEl.classList.remove('hidden');
                confirmBtn.disabled = false;
            });

            xhr.send(formData);
        });

        overlay.querySelector('.cancel-btn').addEventListener('click', function() { overlay.remove(); });
        overlay.addEventListener('click', function(e) { if (e.target === overlay) overlay.remove(); });
    };
})();

// 排序下拉菜单
(function() {
    function getCurrentSort() {
        var sortByInput = document.getElementById('sort-by-input');
        var sortOrderInput = document.getElementById('sort-order-input');
        return {
            sort_by: sortByInput ? sortByInput.value : 'modified_at',
            sort_order: sortOrderInput ? sortOrderInput.value : 'desc'
        };
    }

    function triggerSort(sortBy, sortOrder) {
        var marker = document.getElementById('current-path-marker');
        var path = marker ? (marker.getAttribute('data-path') || '') : '';
        var sortByInput = document.getElementById('sort-by-input');
        var sortOrderInput = document.getElementById('sort-order-input');
        if (sortByInput) sortByInput.value = sortBy;
        if (sortOrderInput) sortOrderInput.value = sortOrder;
        localStorage.setItem('fastpic_sort_by', sortBy);
        localStorage.setItem('fastpic_sort_order', sortOrder);

        var opts = marker ? {
            mode: marker.getAttribute('data-mode') || 'folder',
            sortBy: sortBy,
            sortOrder: sortOrder,
            cols: marker.getAttribute('data-cols') || (document.getElementById('cols-input') ? document.getElementById('cols-input').value : '4')
        } : { sortBy: sortBy, sortOrder: sortOrder };
        htmx.ajax('GET', buildGalleryUrl(path, opts), {target: '#gallery-container', swap: 'innerHTML'});
    }

    document.body.addEventListener('click', function(e) {
        var btn = e.target.closest('#sort-btn');
        var popover = document.getElementById('sort-popover');
        var insidePopover = popover && popover.contains(e.target);

        // 点击排序字段选项
        var fieldOption = e.target.closest('.sort-field-option');
        if (fieldOption && popover && popover.contains(fieldOption)) {
            var current = getCurrentSort();
            triggerSort(fieldOption.getAttribute('data-sort-field'), current.sort_order);
            popover.classList.add('hidden');
            return;
        }

        // 点击排序顺序选项
        var orderOption = e.target.closest('.sort-order-option');
        if (orderOption && popover && popover.contains(orderOption)) {
            var current = getCurrentSort();
            triggerSort(current.sort_by, orderOption.getAttribute('data-sort-order'));
            popover.classList.add('hidden');
            return;
        }

        // 切换排序弹出框
        if (btn) {
            if (popover) {
                popover.classList.toggle('hidden');
                if (!popover.classList.contains('hidden')) {
                    var rect = btn.getBoundingClientRect();
                    var popW = 160;
                    var top = rect.bottom + 8;
                    var left = rect.right - popW;
                    if (left < 8) left = 8;
                    var popH = popover.scrollHeight || 220;
                    if (top + popH > window.innerHeight - 8) {
                        top = rect.top - popH - 8;
                        if (top < 8) top = 8;
                    }
                    popover.style.top = top + 'px';
                    popover.style.left = left + 'px';
                }
            }
        } else if (popover && !popover.classList.contains('hidden') && !insidePopover) {
            popover.classList.add('hidden');
        }
    });
})();

// ---------- 批量选择 & 删除 ----------
(function() {
    window._selectMode = false;
    var _selectedImages = new Set();  // image ids
    var _selectedFolders = new Set(); // folder paths
    var _lastClickedAnchor = null;    // { type, id, path } 用于 Shift 范围选择

    function isSelectMode() { return window._selectMode; }

    function getItemsInOrder() {
        return Array.from(document.querySelectorAll('#gallery-grid .selectable-item'));
    }
    function getItemKey(item) {
        var type = item.getAttribute('data-type');
        if (type === 'folder') return { type: 'folder', id: null, path: item.getAttribute('data-folder-path') };
        return { type: type || 'image', id: parseInt(item.getAttribute('data-image-id')) || null, path: null };
    }
    function findItemIndex(items, key) {
        for (var i = 0; i < items.length; i++) {
            var k = getItemKey(items[i]);
            if (key.type === 'folder' && k.path === key.path) return i;
            if (key.type !== 'folder' && k.id === key.id) return i;
        }
        return -1;
    }

    function enterSelectMode() {
        window._selectMode = true;
        var btn = document.getElementById('select-mode-btn');
        if (btn) { btn.classList.add('bg-blue-100', 'text-blue-600'); btn.classList.remove('hover:bg-slate-100', 'hover:text-slate-700'); }
        var selAllBtn = document.getElementById('select-all-btn');
        if (selAllBtn) selAllBtn.classList.remove('hidden');
        // 显示所有复选框
        document.querySelectorAll('.select-checkbox').forEach(function(cb) { cb.classList.remove('hidden'); });
    }

    function exitSelectMode() {
        window._selectMode = false;
        _selectedImages.clear();
        _selectedFolders.clear();
        _lastClickedAnchor = null;
        var btn = document.getElementById('select-mode-btn');
        if (btn) { btn.classList.remove('bg-blue-100', 'text-blue-600'); btn.classList.add('hover:bg-slate-100', 'hover:text-slate-700'); }
        var selAllBtn = document.getElementById('select-all-btn');
        if (selAllBtn) selAllBtn.classList.add('hidden');
        var delBtn = document.getElementById('delete-selected-btn');
        if (delBtn) delBtn.classList.add('hidden');
        var moveBtn = document.getElementById('move-selected-btn');
        if (moveBtn) moveBtn.classList.add('hidden');
        var downloadBtn = document.getElementById('download-selected-btn');
        if (downloadBtn) downloadBtn.classList.add('hidden');
        // 隐藏底部固定操作栏
        var fab = document.getElementById('select-mode-fab');
        if (fab) fab.classList.add('hidden');
        var scrollEl = document.getElementById('scroll-container');
        if (scrollEl) scrollEl.classList.remove('pb-24');
        // 隐藏复选框，取消选中效果
        document.querySelectorAll('.select-checkbox').forEach(function(cb) { cb.classList.add('hidden'); });
        document.querySelectorAll('.selectable-item').forEach(function(item) { uncheckItem(item); });
    }

    window.toggleSelectMode = function() {
        if (isSelectMode()) { exitSelectMode(); } else { enterSelectMode(); }
    };
    window.exitSelectMode = exitSelectMode;
    window.getSelectedFolders = function() { return Array.from(_selectedFolders); };
    window.getSelectedImages = function() { return Array.from(_selectedImages); };

    function checkItem(item) {
        item.setAttribute('data-selected', 'true');
        var overlay = item.querySelector('.select-overlay');
        if (overlay) overlay.classList.remove('hidden');
        var cbDiv = item.querySelector('.select-checkbox > div');
        if (cbDiv) { cbDiv.classList.add('bg-blue-500', 'border-blue-500'); cbDiv.classList.remove('bg-white/80'); }
        var icon = item.querySelector('.check-icon');
        if (icon) icon.classList.remove('hidden');
    }

    function uncheckItem(item) {
        item.removeAttribute('data-selected');
        var overlay = item.querySelector('.select-overlay');
        if (overlay) overlay.classList.add('hidden');
        var cbDiv = item.querySelector('.select-checkbox > div');
        if (cbDiv) { cbDiv.classList.remove('bg-blue-500', 'border-blue-500'); cbDiv.classList.add('bg-white/80'); }
        var icon = item.querySelector('.check-icon');
        if (icon) icon.classList.add('hidden');
    }

    function updateDeleteButton() {
        var total = _selectedImages.size + _selectedFolders.size;
        var imgCount = _selectedImages.size;
        var folderCount = _selectedFolders.size;
        var delBtn = document.getElementById('delete-selected-btn');
        var moveBtn = document.getElementById('move-selected-btn');
        var downloadBtn = document.getElementById('download-selected-btn');
        var fabMoveBtn = document.getElementById('select-mode-fab-move');
        var fabRenameBtn = document.getElementById('select-mode-fab-rename');
        var countText = document.getElementById('delete-count-text');
        var fab = document.getElementById('select-mode-fab');
        var fabCount = document.getElementById('select-mode-fab-count');
        var scrollEl = document.getElementById('scroll-container');
        if (total > 0) {
            if (delBtn) delBtn.classList.remove('hidden');
            if (downloadBtn) downloadBtn.classList.remove('hidden');
            if (countText) countText.textContent = '删除 (' + total + ')';
            if (moveBtn) (total > 0 ? moveBtn.classList.remove('hidden') : moveBtn.classList.add('hidden'));
            if (fabMoveBtn) (total > 0 ? fabMoveBtn.classList.remove('hidden') : fabMoveBtn.classList.add('hidden'));
            if (fabRenameBtn) {
                var showRename = (folderCount === 1 && imgCount === 0) || (imgCount === 1 && folderCount === 0) || (folderCount > 1 && imgCount === 0) || (imgCount > 1 && folderCount === 0);
                showRename ? fabRenameBtn.classList.remove('hidden') : fabRenameBtn.classList.add('hidden');
            }
            if (fab) fab.classList.remove('hidden');
            if (fabCount) fabCount.textContent = '已选 ' + total + ' 项';
            if (scrollEl) scrollEl.classList.add('pb-24');
        } else {
            if (delBtn) delBtn.classList.add('hidden');
            if (downloadBtn) downloadBtn.classList.add('hidden');
            if (moveBtn) moveBtn.classList.add('hidden');
            if (fabMoveBtn) fabMoveBtn.classList.add('hidden');
            if (fabRenameBtn) fabRenameBtn.classList.add('hidden');
            if (fab) fab.classList.add('hidden');
            if (scrollEl) scrollEl.classList.remove('pb-24');
        }
    }

    function applySelectionToItem(item, select) {
        var type = item.getAttribute('data-type');
        if (select) {
            checkItem(item);
            if (type === 'image' || type === 'video') _selectedImages.add(parseInt(item.getAttribute('data-image-id')));
            if (type === 'folder') _selectedFolders.add(item.getAttribute('data-folder-path'));
        } else {
            uncheckItem(item);
            if (type === 'image' || type === 'video') _selectedImages.delete(parseInt(item.getAttribute('data-image-id')));
            if (type === 'folder') _selectedFolders.delete(item.getAttribute('data-folder-path'));
        }
    }

    window.handleItemClick = function(event, item) {
        if (!isSelectMode()) {
            // 非选择模式：文件夹正常导航，图片正常预览
            return;
        }
        // 选择模式下阻止所有导航/链接
        event.preventDefault();
        event.stopPropagation();

        var type = item.getAttribute('data-type');
        var currentKey = getItemKey(item);

        // Alt: 反选
        if (event.altKey) {
            var items = getItemsInOrder();
            items.forEach(function(it) {
                var sel = it.getAttribute('data-selected') === 'true';
                applySelectionToItem(it, !sel);
            });
            _lastClickedAnchor = currentKey;
            updateDeleteButton();
            return;
        }

        // Shift: 范围选择（替换为锚点到当前的区间）
        if (event.shiftKey) {
            var items = getItemsInOrder();
            var anchorIdx = _lastClickedAnchor ? findItemIndex(items, _lastClickedAnchor) : -1;
            var currentIdx = findItemIndex(items, currentKey);
            if (currentIdx < 0) { _lastClickedAnchor = currentKey; updateDeleteButton(); return; }
            if (anchorIdx < 0) {
                // 无锚点：仅选中当前项
                applySelectionToItem(item, true);
            } else {
                var lo = Math.min(anchorIdx, currentIdx);
                var hi = Math.max(anchorIdx, currentIdx);
                // 先清空当前可见项的选择，再选中范围
                items.forEach(function(it) { applySelectionToItem(it, false); });
                for (var i = lo; i <= hi; i++) {
                    applySelectionToItem(items[i], true);
                }
            }
            _lastClickedAnchor = currentKey;
            updateDeleteButton();
            return;
        }

        // Ctrl/Meta 或无：切换当前项
        var isSelected = item.getAttribute('data-selected') === 'true';
        applySelectionToItem(item, !isSelected);
        _lastClickedAnchor = currentKey;
        updateDeleteButton();
    };

    // 捕获阶段拦截：选择模式下点击 selectable-item（含文件夹内的链接）时，优先处理选择，阻止导航/HTMX
    document.addEventListener('click', function(e) {
        if (!isSelectMode()) return;
        var item = e.target.closest('.selectable-item');
        if (!item) return;
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();
        window.handleItemClick(e, item);
    }, true);

    window.toggleSelectAll = function() {
        var items = document.querySelectorAll('.selectable-item');
        // 判断是全选还是取消全选
        var allSelected = true;
        items.forEach(function(item) {
            if (item.getAttribute('data-selected') !== 'true') allSelected = false;
        });
        items.forEach(function(item) {
            var type = item.getAttribute('data-type');
            if (allSelected) {
                uncheckItem(item);
                if (type === 'image' || type === 'video') _selectedImages.delete(parseInt(item.getAttribute('data-image-id')));
                if (type === 'folder') _selectedFolders.delete(item.getAttribute('data-folder-path'));
            } else {
                checkItem(item);
                if (type === 'image' || type === 'video') _selectedImages.add(parseInt(item.getAttribute('data-image-id')));
                if (type === 'folder') _selectedFolders.add(item.getAttribute('data-folder-path'));
            }
        });
        updateDeleteButton();
    };

    window.confirmDeleteSelected = function() {
        var imgCount = _selectedImages.size;
        var folderCount = _selectedFolders.size;
        if (imgCount === 0 && folderCount === 0) return;

        var msg = '确定要删除以下内容吗？\n\n';
        if (folderCount > 0) msg += '• ' + folderCount + ' 个文件夹（包含其中所有图片）\n';
        if (imgCount > 0) msg += '• ' + imgCount + ' 张图片\n';
        msg += '\n此操作不可恢复！';

        // 显示自定义确认对话框
        showDeleteConfirmDialog(msg, function() {
            executeDelete(Array.from(_selectedImages), Array.from(_selectedFolders));
        });
    };

    var _beforeUnloadHandler = function(e) {
        e.preventDefault();
        e.returnValue = '';
    };
    var OPERATION_STORAGE_KEY = 'fastpic_operation';
    var OPERATION_MAX_AGE_MS = 5 * 60 * 1000; // 5 分钟

    window.showOperationLoading = function(message) {
        var msg = message || '请稍候...';
        var el = document.getElementById('operation-loading-overlay');
        var msgEl = document.getElementById('operation-loading-message');
        if (msgEl) msgEl.textContent = msg;
        if (el) el.classList.remove('hidden');
        window.addEventListener('beforeunload', _beforeUnloadHandler);
        try {
            localStorage.setItem(OPERATION_STORAGE_KEY, JSON.stringify({ type: msg, ts: Date.now() }));
        } catch (err) {}
    };
    window.hideOperationLoading = function() {
        var el = document.getElementById('operation-loading-overlay');
        if (el) el.classList.add('hidden');
        window.removeEventListener('beforeunload', _beforeUnloadHandler);
        try {
            localStorage.removeItem(OPERATION_STORAGE_KEY);
        } catch (err) {}
    };

    window.downloadSelected = function() {
        var imgCount = _selectedImages.size;
        var folderCount = _selectedFolders.size;
        if (imgCount === 0 && folderCount === 0) return;
        if (imgCount === 1 && folderCount === 0) {
            var id = Array.from(_selectedImages)[0];
            window.location.href = '/api/download/image?id=' + encodeURIComponent(id);
            return;
        }
        var btn = document.getElementById('select-mode-fab-download');
        if (btn) btn.disabled = true;
        if (window.showOperationLoading) window.showOperationLoading('正在打包，请稍候...');
        fetch('/api/download/zip', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_ids: Array.from(_selectedImages),
                folder_paths: Array.from(_selectedFolders)
            })
        }).then(function(r) {
            if (!r.ok) return r.json().then(function(d) { throw new Error(d.detail || '下载失败'); });
            return r.blob();
        }).then(function(blob) {
            var url = URL.createObjectURL(blob);
            var a = document.createElement('a');
            a.href = url;
            a.download = 'download.zip';
            a.style.display = 'none';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }).catch(function(err) {
            if (typeof showToast === 'function') showToast(err.message || '下载失败', 'error');
        }).finally(function() {
            if (btn) btn.disabled = false;
            if (window.hideOperationLoading) window.hideOperationLoading();
        });
    };

    function showDeleteConfirmDialog(message, onConfirm) {
        // 移除之前的对话框（如果有）
        var old = document.getElementById('delete-confirm-dialog');
        if (old) old.remove();

        var overlay = document.createElement('div');
        overlay.id = 'delete-confirm-dialog';
        overlay.className = 'fixed inset-0 z-[100] flex items-center justify-center bg-black/40';
        overlay.innerHTML =
            '<div class="bg-white rounded-xl shadow-2xl max-w-md w-full mx-4 p-6">' +
                '<div class="flex items-center gap-3 mb-4">' +
                    '<div class="w-10 h-10 rounded-full bg-red-100 flex items-center justify-center flex-shrink-0">' +
                        '<svg class="w-5 h-5 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-width="2" stroke-linecap="round" stroke-linejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4.5c-.77-.833-2.694-.833-3.464 0L3.34 16.5c-.77.833.192 2.5 1.732 2.5z"/></svg>' +
                    '</div>' +
                    '<h3 class="text-lg font-semibold text-slate-800">确认删除</h3>' +
                '</div>' +
                '<pre class="text-sm text-slate-600 mb-6 whitespace-pre-wrap font-sans">' + message.replace(/</g, '&lt;') + '</pre>' +
                '<div class="flex justify-end gap-3">' +
                    '<button type="button" class="cancel-btn px-4 py-2 rounded-lg border border-slate-300 text-slate-700 hover:bg-slate-50 text-sm font-medium transition-colors">取消</button>' +
                    '<button type="button" class="confirm-btn px-4 py-2 rounded-lg bg-red-500 hover:bg-red-600 text-white text-sm font-medium transition-colors">确认删除</button>' +
                '</div>' +
            '</div>';

        document.body.appendChild(overlay);

        overlay.querySelector('.cancel-btn').addEventListener('click', function() { overlay.remove(); });
        overlay.querySelector('.confirm-btn').addEventListener('click', function() {
            overlay.remove();
            onConfirm();
        });
        overlay.addEventListener('click', function(e) { if (e.target === overlay) overlay.remove(); });
    }

    function executeDelete(imageIds, folderPaths) {
        var delBtn = document.getElementById('delete-selected-btn');
        var fabDelBtn = document.getElementById('select-mode-fab-delete');
        if (window.showOperationLoading) window.showOperationLoading('正在删除，请稍候...');
        if (delBtn) {
            delBtn.disabled = true;
            var countText = delBtn.querySelector('#delete-count-text');
            if (countText) countText.textContent = '删除中...';
        }
        if (fabDelBtn) {
            fabDelBtn.disabled = true;
            fabDelBtn.innerHTML = '<span>删除中...</span>';
        }

        var promises = [];

        if (imageIds.length > 0) {
            promises.push(
                fetch('/api/delete-images', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ids: imageIds})
                }).then(function(r) { return r.json(); })
            );
        }

        if (folderPaths.length > 0) {
            promises.push(
                fetch('/api/delete-folders', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({paths: folderPaths})
                }).then(function(r) { return r.json(); })
            );
        }

        Promise.all(promises).then(function(results) {
            // 退出选择模式
            exitSelectMode();
            refreshGallery();
        }).catch(function(err) {
            console.error('删除失败:', err);
            if (typeof showToast === 'function') showToast('删除失败，请查看控制台日志', 'error');
            if (delBtn) delBtn.disabled = false;
            if (fabDelBtn) {
                fabDelBtn.disabled = false;
                fabDelBtn.innerHTML = '<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-width="2" stroke-linecap="round" stroke-linejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/></svg>\n            删除';
            }
            updateDeleteButton();
        }).finally(function() {
            if (window.hideOperationLoading) window.hideOperationLoading();
            if (delBtn) delBtn.disabled = false;
            var countText = delBtn ? delBtn.querySelector('#delete-count-text') : null;
            if (countText) countText.textContent = '删除';
            if (fabDelBtn) {
                fabDelBtn.disabled = false;
                fabDelBtn.innerHTML = '<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-width="2" stroke-linecap="round" stroke-linejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/></svg>\n            删除';
            }
        });
    }

    function getCurrentPathForMove() {
        var marker = document.getElementById('current-path-marker');
        var path = marker ? (marker.getAttribute('data-path') || '') : '';
        var pathInput = document.querySelector('[name=path]');
        if (pathInput) path = pathInput.value || path;
        return path || '';
    }

    window.showMoveDialog = function(imageIdsOrNull, options) {
        var imgIds = imageIdsOrNull !== undefined && imageIdsOrNull !== null
            ? (Array.isArray(imageIdsOrNull) ? imageIdsOrNull : [imageIdsOrNull])
            : Array.from(_selectedImages);
        var folderPaths = (options && options.folderPaths) ? options.folderPaths : Array.from(_selectedFolders);
        if (imgIds.length === 0 && folderPaths.length === 0) return;
        var fromModal = options && options.fromModal;

        var titleText = '移动';
        var descParts = [];
        if (imgIds.length > 0) descParts.push(imgIds.length + ' 张图片');
        if (folderPaths.length > 0) descParts.push(folderPaths.length + ' 个文件夹');
        var descText = '将 ' + descParts.join(' 和 ') + ' 移动到：';

        var old = document.getElementById('move-dialog');
        if (old) old.remove();

        var overlay = document.createElement('div');
        overlay.id = 'move-dialog';
        overlay.className = 'fixed inset-0 z-[100] flex items-center justify-center bg-black/40';
        overlay.innerHTML =
            '<div class="bg-white rounded-xl shadow-2xl max-w-md w-full mx-4 max-h-[85vh] flex flex-col">' +
                '<div class="p-4 border-b border-slate-200 flex-shrink-0">' +
                    '<div class="flex items-center gap-3 mb-2">' +
                        '<div class="w-10 h-10 rounded-full bg-blue-100 flex items-center justify-center flex-shrink-0">' +
                            '<svg class="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-width="2" stroke-linecap="round" stroke-linejoin="round" d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4"/></svg>' +
                        '</div>' +
                        '<h3 class="text-lg font-semibold text-slate-800">' + titleText + '</h3>' +
                    '</div>' +
                    '<div class="text-sm text-slate-500 move-dialog-desc">' + descText + '</div>' +
                    '<nav id="move-breadcrumb" class="mt-2 flex items-center gap-1 text-sm text-slate-600 flex-wrap"></nav>' +
                '</div>' +
                '<div id="move-folder-list" class="flex-1 overflow-y-auto p-4 min-h-0"></div>' +
                '<div class="p-4 border-t border-slate-200 flex-shrink-0 flex justify-end gap-2">' +
                    '<button type="button" class="move-cancel-btn px-4 py-2 rounded-lg border border-slate-300 text-slate-700 hover:bg-slate-50 text-sm font-medium transition-colors">取消</button>' +
                    '<button type="button" class="move-confirm-btn px-4 py-2 rounded-lg bg-blue-500 hover:bg-blue-600 text-white text-sm font-medium transition-colors">移动到这里</button>' +
                '</div>' +
                '<div id="move-error" class="hidden px-4 pb-4 text-sm text-red-500"></div>' +
            '</div>';

        document.body.appendChild(overlay);

        var breadcrumbEl = overlay.querySelector('#move-breadcrumb');
        var listEl = overlay.querySelector('#move-folder-list');
        var confirmBtn = overlay.querySelector('.move-confirm-btn');
        var errorEl = overlay.querySelector('#move-error');

        var newFolderRowHtml = '<div id="move-new-folder-row" class="mb-3 flex-shrink-0">' +
            '<div id="move-new-folder-btn" class="flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-slate-50 cursor-pointer transition-colors text-blue-600">' +
            '<svg class="w-5 h-5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-width="2" stroke-linecap="round" stroke-linejoin="round" d="M12 6v6m0 0v6m0-6h6m-6 0H6"/></svg>' +
            '<span class="text-sm font-medium">新建文件夹</span>' +
            '</div>' +
            '<div id="move-new-folder-input-row" class="hidden flex items-center gap-2 mt-2">' +
            '<input type="text" id="move-new-folder-input" placeholder="输入文件夹名称" class="flex-1 px-3 py-2 border border-slate-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500">' +
            '<button type="button" class="move-new-folder-confirm px-3 py-2 rounded-lg bg-blue-500 hover:bg-blue-600 text-white text-sm font-medium transition-colors">创建</button>' +
            '<button type="button" class="move-new-folder-cancel px-3 py-2 rounded-lg border border-slate-300 text-slate-700 hover:bg-slate-50 text-sm font-medium transition-colors">取消</button>' +
            '</div>' +
            '<div id="move-new-folder-error" class="hidden mt-1 text-sm text-red-500"></div>' +
            '</div>';

        var browser = createFolderBrowser(breadcrumbEl, listEl, {
            initialPath: (options && options.initialPath !== undefined) ? options.initialPath : getCurrentPathForMove(),
            itemClass: 'move-folder-item',
            navClass: 'move-nav-link',
            prependHtml: newFolderRowHtml,
            afterRender: function(b) {
                var newFolderBtn = listEl.querySelector('#move-new-folder-btn');
                var inputRow = listEl.querySelector('#move-new-folder-input-row');
                var newFolderInput = listEl.querySelector('#move-new-folder-input');
                var newFolderConfirm = listEl.querySelector('.move-new-folder-confirm');
                var newFolderCancel = listEl.querySelector('.move-new-folder-cancel');
                var newFolderError = listEl.querySelector('#move-new-folder-error');
                if (newFolderBtn) {
                    newFolderBtn.onclick = function() {
                        newFolderBtn.classList.add('hidden');
                        inputRow.classList.remove('hidden');
                        inputRow.classList.add('flex');
                        if (newFolderError) newFolderError.classList.add('hidden');
                        if (newFolderInput) { newFolderInput.value = ''; setTimeout(function() { newFolderInput.focus(); }, 50); }
                    };
                }
                if (newFolderCancel) {
                    newFolderCancel.onclick = function() {
                        inputRow.classList.add('hidden');
                        inputRow.classList.remove('flex');
                        if (newFolderBtn) newFolderBtn.classList.remove('hidden');
                        if (newFolderError) newFolderError.classList.add('hidden');
                    };
                }
                function doCreateNewFolder() {
                    var name = newFolderInput ? newFolderInput.value.trim() : '';
                    if (!name) { if (newFolderError) { newFolderError.textContent = '名称不能为空'; newFolderError.classList.remove('hidden'); } return; }
                    if (/[\/\\]/.test(name) || name.includes('..')) { if (newFolderError) { newFolderError.textContent = '名称不能包含 / \\ 或 ..'; newFolderError.classList.remove('hidden'); } return; }
                    if (newFolderConfirm) newFolderConfirm.disabled = true;
                    if (newFolderConfirm) newFolderConfirm.textContent = '创建中...';
                    fetch('/api/create-folder', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({path: b.getPath(), name: name})
                    }).then(function(r) { return r.json(); }).then(function(data) {
                        if (data.ok) {
                            if (inputRow) { inputRow.classList.add('hidden'); inputRow.classList.remove('flex'); }
                            if (newFolderBtn) newFolderBtn.classList.remove('hidden');
                            if (newFolderError) newFolderError.classList.add('hidden');
                            if (newFolderConfirm) { newFolderConfirm.disabled = false; newFolderConfirm.textContent = '创建'; }
                            b.refresh();
                        } else {
                            if (newFolderError) { newFolderError.textContent = data.error || '创建失败'; newFolderError.classList.remove('hidden'); }
                            if (newFolderConfirm) { newFolderConfirm.disabled = false; newFolderConfirm.textContent = '创建'; }
                        }
                    }).catch(function() {
                        if (newFolderError) { newFolderError.textContent = '请求失败'; newFolderError.classList.remove('hidden'); }
                        if (newFolderConfirm) { newFolderConfirm.disabled = false; newFolderConfirm.textContent = '创建'; }
                    });
                }
                if (newFolderConfirm) newFolderConfirm.onclick = doCreateNewFolder;
                if (newFolderInput) {
                    newFolderInput.onkeydown = function(e) {
                        if (e.key === 'Enter') { e.preventDefault(); doCreateNewFolder(); }
                        if (e.key === 'Escape') { if (newFolderCancel) newFolderCancel.click(); }
                    };
                }
            }
        });
        browser.init();

        function doMove() {
            errorEl.classList.add('hidden');
            confirmBtn.disabled = true;
            confirmBtn.textContent = '移动中...';
            if (window.showOperationLoading) window.showOperationLoading('正在移动，请稍候...');

            var currentPath = browser.getPath();
            var promises = [];
            if (imgIds.length > 0) {
                promises.push(
                    fetch('/api/move-images', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ids: imgIds, target_path: currentPath})
                    }).then(function(r) { return r.json(); })
                );
            }
            if (folderPaths.length > 0) {
                promises.push(
                    fetch('/api/move-folders', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({paths: folderPaths, target_path: currentPath})
                    }).then(function(r) { return r.json(); })
                );
            }

            Promise.all(promises).then(function(results) {
                var totalMoved = 0;
                var allErrors = [];
                results.forEach(function(data) {
                    totalMoved += (data.moved || 0);
                    if (data.errors && data.errors.length) allErrors = allErrors.concat(data.errors);
                });

                overlay.remove();
                if (totalMoved > 0) {
                    if (fromModal && options && options.onSuccess) {
                        options.onSuccess();
                    } else {
                        imgIds.forEach(function(id) { _selectedImages.delete(id); });
                        folderPaths.forEach(function(p) { _selectedFolders.delete(p); });
                        updateDeleteButton();
                        if (_selectedImages.size === 0 && _selectedFolders.size === 0) { exitSelectMode(); }
                    }
                    if (allErrors.length > 0) {
                        if (typeof showToast === 'function') showToast('已移动 ' + totalMoved + ' 项，失败 ' + allErrors.length + ' 项：' + allErrors.join('；'), 'error');
                    }
                    if (!fromModal) {
                        refreshGallery();
                    }
                } else {
                    if (allErrors.length > 0) {
                        errorEl.textContent = allErrors.join('；');
                        errorEl.classList.remove('hidden');
                    } else {
                        errorEl.textContent = '所选内容已在目标位置';
                        errorEl.classList.remove('hidden');
                    }
                    confirmBtn.disabled = false;
                    confirmBtn.textContent = '移动到这里';
                }
            }).catch(function(err) {
                console.error('移动失败:', err);
                if (typeof showToast === 'function') showToast('移动失败', 'error');
                errorEl.textContent = '请求失败: ' + err;
                errorEl.classList.remove('hidden');
                confirmBtn.disabled = false;
                confirmBtn.textContent = '移动到这里';
            }).finally(function() {
                if (window.hideOperationLoading) window.hideOperationLoading();
            });
        }

        overlay.querySelector('.move-cancel-btn').addEventListener('click', function() { overlay.remove(); });
        confirmBtn.addEventListener('click', doMove);
        overlay.addEventListener('click', function(e) { if (e.target === overlay) overlay.remove(); });
    };

    window.showRenameDialogForCurrentImage = function() {
        if (modalImageIds.length === 0 || modalIndex >= modalImageIds.length) return;
        var id = modalImageIds[modalIndex];
        if (!id) return;
        window.showRenameImageDialog && window.showRenameImageDialog(id);
    };

    window.showMoveDialogForCurrentImage = function() {
        if (modalImageIds.length === 0 || modalIndex >= modalImageIds.length) return;
        var id = modalImageIds[modalIndex];
        if (!id) return;
        window.showMoveDialog([id], {
            fromModal: true,
            initialPath: getCurrentPathForMove(),
            onSuccess: function() {
                modalImages.splice(modalIndex, 1);
                modalImageIds.splice(modalIndex, 1);
                modalMediaTypes.splice(modalIndex, 1);
                if (modalImages.length === 0) {
                    closeModalAndStopSlideshow();
                } else {
                    if (modalIndex >= modalImages.length) modalIndex = modalImages.length - 1;
                    _showModalContent(modalImages[modalIndex], modalMediaTypes[modalIndex]);
                    _updateModalImageCounter();
                    refreshImageInfoIfVisible();
                }
                refreshGalleryFromModal();
            }
        });
    };

    function refreshGallery() {
        if (window.galleryPathCache) window.galleryPathCache.clear();
        if (window.folderImagesCache) window.folderImagesCache.clear();
        var marker = document.getElementById('current-path-marker');
        var path = marker ? (marker.getAttribute('data-path') || '') : '';
        var colsInput = document.getElementById('cols-input');
        var opts = marker ? {
            mode: marker.getAttribute('data-mode') || 'folder',
            sortBy: marker.getAttribute('data-sort-by') || 'modified_at',
            sortOrder: marker.getAttribute('data-sort-order') || 'desc',
            cols: (colsInput && colsInput.value) ? colsInput.value : (marker.getAttribute('data-cols') || '4')
        } : undefined;
        htmx.ajax('GET', buildGalleryUrl(path, opts), {target: '#gallery-container', swap: 'innerHTML'});
    }
    window.refreshGallery = refreshGallery;

    // HTMX 加载新内容后，如果处于选择模式，需要更新新内容的复选框状态
    document.body.addEventListener('htmx:afterSettle', function(ev) {
        if (!isSelectMode()) return;
        if (ev.detail.target.id === 'gallery-container' || ev.detail.target.id === 'gallery-grid') {
            // 重新显示复选框
            document.querySelectorAll('.select-checkbox').forEach(function(cb) { cb.classList.remove('hidden'); });
        }
    });
})();

// ---------- 筛选面板 ----------
(function() {
    // 当前筛选状态（从 gallery 片段中读取）
    var _filterState = {
        filter_filename: '',
        filter_size_min: '',
        filter_size_max: '',
        filter_date_from: '',
        filter_date_to: '',
        filter_tag: ''
    };

    function readFilterState() {
        var fn = document.getElementById('filter-filename-input');
        var sMin = document.getElementById('filter-size-min-input');
        var sMax = document.getElementById('filter-size-max-input');
        var dFrom = document.getElementById('filter-date-from-input');
        var dTo = document.getElementById('filter-date-to-input');
        var tagInput = document.getElementById('filter-tag-input');
        return {
            filter_filename: fn ? fn.value.trim() : '',
            filter_size_min: sMin ? sMin.value : '',
            filter_size_max: sMax ? sMax.value : '',
            filter_date_from: dFrom ? dFrom.value : '',
            filter_date_to: dTo ? dTo.value : '',
            filter_tag: tagInput ? tagInput.value.trim() : ''
        };
    }

    function triggerFilteredGallery(filters) {
        var marker = document.getElementById('current-path-marker');
        var path = marker ? (marker.getAttribute('data-path') || '') : '';
        var opts = marker ? {
            mode: marker.getAttribute('data-mode') || 'folder',
            sortBy: marker.getAttribute('data-sort-by') || 'modified_at',
            sortOrder: marker.getAttribute('data-sort-order') || 'desc',
            cols: marker.getAttribute('data-cols') || (document.getElementById('cols-input') ? document.getElementById('cols-input').value : '4'),
            filters: filters
        } : { filters: filters };
        htmx.ajax('GET', buildGalleryUrl(path, opts), {target: '#gallery-container', swap: 'innerHTML'});
    }

    function positionFilterPopover(btn, popover) {
        var rect = btn.getBoundingClientRect();
        var popW = 320; // w-80 = 20rem = 320px
        // 优先放在按钮下方、右对齐
        var top = rect.bottom + 8;
        var left = rect.right - popW;
        // 防止左侧溢出
        if (left < 8) left = 8;
        // 防止底部溢出：如果放不下就放到按钮上方
        var popH = popover.scrollHeight || 340;
        if (top + popH > window.innerHeight - 8) {
            top = rect.top - popH - 8;
            if (top < 8) top = 8;
        }
        popover.style.top = top + 'px';
        popover.style.left = left + 'px';
    }

    // 事件委托：点击筛选按钮打开/关闭面板
    document.body.addEventListener('click', function(e) {
        var filterBtn = e.target.closest('#filter-btn');
        var filterPopover = document.getElementById('filter-popover');
        var insidePopover = filterPopover && filterPopover.contains(e.target);

        if (filterBtn) {
            if (filterPopover) {
                filterPopover.classList.toggle('hidden');
                if (!filterPopover.classList.contains('hidden')) {
                    positionFilterPopover(filterBtn, filterPopover);
                }
            }
            return;
        }

        // 点击应用筛选
        if (e.target.closest('#filter-apply-btn')) {
            var filters = readFilterState();
            _filterState = filters;
            if (filterPopover) filterPopover.classList.add('hidden');
            triggerFilteredGallery(filters);
            return;
        }

        // 点击清除全部
        if (e.target.closest('#filter-clear-btn')) {
            // 重置所有输入
            var fn = document.getElementById('filter-filename-input');
            var tagInput = document.getElementById('filter-tag-input');
            var sMin = document.getElementById('filter-size-min-input');
            var sMax = document.getElementById('filter-size-max-input');
            var dFrom = document.getElementById('filter-date-from-input');
            var dTo = document.getElementById('filter-date-to-input');
            if (fn) fn.value = '';
            if (sMin) sMin.value = '';
            if (sMax) sMax.value = '';
            if (dFrom) dFrom.value = '';
            if (dTo) dTo.value = '';
            if (tagInput) tagInput.value = '';
            _filterState = {filter_filename: '', filter_size_min: '', filter_size_max: '', filter_date_from: '', filter_date_to: '', filter_tag: ''};
            if (filterPopover) filterPopover.classList.add('hidden');
            triggerFilteredGallery(_filterState);
            return;
        }

        // 点击外部关闭面板
        if (filterPopover && !filterPopover.classList.contains('hidden') && !insidePopover) {
            filterPopover.classList.add('hidden');
        }
    });

    // 回车键应用筛选
    document.body.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') {
            var filterPopover = document.getElementById('filter-popover');
            if (filterPopover && !filterPopover.classList.contains('hidden')) {
                var isInsideFilter = e.target.closest('#filter-popover');
                if (isInsideFilter) {
                    e.preventDefault();
                    var filters = readFilterState();
                    _filterState = filters;
                    filterPopover.classList.add('hidden');
                    triggerFilteredGallery(filters);
                }
            }
        }
    });

    // 暴露过滤状态到全局（其他逻辑可能需要）
    window.getFilterState = function() { return _filterState; };
})();

// ---------- 顶部目录搜索 ----------
(function() {
    var searchInput = document.getElementById('dir-search-input');
    var dropdown = document.getElementById('dir-search-dropdown');

    function doSearch(query) {
        if (!query.trim()) {
            dropdown.classList.add('hidden');
            dropdown.innerHTML = '';
            return;
        }
        fetch('/api/search-dirs?q=' + encodeURIComponent(query) + '&limit=20')
            .then(function(r) { return r.json(); })
            .then(function(data) {
                if (!data.dirs || data.dirs.length === 0) {
                    dropdown.innerHTML = '<div class="px-4 py-3 text-sm text-slate-400">未找到匹配的文件夹</div>';
                    dropdown.classList.remove('hidden');
                    return;
                }
                var html = '';
                for (var i = 0; i < data.dirs.length; i++) {
                    var d = data.dirs[i];
                    // 高亮匹配部分
                    var displayPath = highlightMatch(d.path, query);
                    html += '<button type="button" class="dir-search-item w-full text-left px-4 py-2.5 hover:bg-blue-50 transition-colors flex items-center gap-3 border-b border-slate-50 last:border-0" data-path="' + escapeAttr(d.path) + '">' +
                        '<svg class="w-4 h-4 text-slate-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-width="2" stroke-linecap="round" stroke-linejoin="round" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"/></svg>' +
                        '<span class="text-sm text-slate-700 truncate flex-1">' + displayPath + '</span>' +
                        '<span class="text-xs text-slate-400 flex-shrink-0">' + d.image_count + ' 张</span>' +
                    '</button>';
                }
                dropdown.innerHTML = html;
                dropdown.classList.remove('hidden');
            })
            .catch(function() {
                dropdown.classList.add('hidden');
            });
    }

    function highlightMatch(path, query) {
        var ql = query.toLowerCase();
        var pl = path.toLowerCase();
        var idx = pl.indexOf(ql);
        if (idx === -1) return escapeHtml(path);
        return escapeHtml(path.substring(0, idx)) +
            '<span class="text-blue-600 font-medium">' + escapeHtml(path.substring(idx, idx + query.length)) + '</span>' +
            escapeHtml(path.substring(idx + query.length));
    }

    function navigateToDir(dirPath) {
        dropdown.classList.add('hidden');
        searchInput.value = '';
        document.querySelector('[name=path]').value = dirPath;
        htmx.ajax('GET', buildGalleryUrl(dirPath, { filters: {} }), {target: '#gallery-container', swap: 'innerHTML'});
    }

    if (searchInput) {
        var debouncedSearch = debounce(doSearch, 250);
        searchInput.addEventListener('input', function() {
            debouncedSearch(this.value);
        });

        searchInput.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                dropdown.classList.add('hidden');
                this.blur();
                return;
            }
            if (e.key === 'Enter') {
                var active = dropdown.querySelector('.dir-search-item.bg-blue-50');
                var target = active || dropdown.querySelector('.dir-search-item');
                if (target) {
                    e.preventDefault();
                    navigateToDir(target.getAttribute('data-path'));
                }
                return;
            }
            if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
                e.preventDefault();
                var items = dropdown.querySelectorAll('.dir-search-item');
                if (items.length === 0) return;
                var active = dropdown.querySelector('.dir-search-item.bg-blue-50');
                var idx = -1;
                for (var i = 0; i < items.length; i++) {
                    if (items[i] === active) { idx = i; break; }
                }
                if (active) active.classList.remove('bg-blue-50');
                if (e.key === 'ArrowDown') idx = (idx + 1) % items.length;
                else idx = idx <= 0 ? items.length - 1 : idx - 1;
                items[idx].classList.add('bg-blue-50');
                items[idx].scrollIntoView({block: 'nearest'});
            }
        });
    }

    // 点击下拉结果
    if (dropdown) {
        dropdown.addEventListener('click', function(e) {
            var item = e.target.closest('.dir-search-item');
            if (item) {
                navigateToDir(item.getAttribute('data-path'));
            }
        });
    }

    // 点击外部关闭下拉
    document.addEventListener('click', function(e) {
        var wrapper = document.getElementById('dir-search-wrapper');
        if (wrapper && !wrapper.contains(e.target)) {
            dropdown.classList.add('hidden');
        }
    });
})();

// popstate：浏览器后退/前进时加载对应 gallery；若大图模式已打开则先关闭
window.addEventListener('popstate', function(ev) {
    if (location.pathname !== '/' && location.pathname !== '') return;
    var modal = document.getElementById('modal');
    if (modal && modal.classList.contains('modal-open')) {
        closeModalAndStopSlideshow();
    }
    var path = getPathFromUrl();
    window._restoringFromPopstate = true;
    var url = buildGalleryUrl(path);
    htmx.ajax('GET', url, {target: '#gallery-container', swap: 'innerHTML'});
});

// HTMX 交换体验优化：加载指示器延迟显示、容器淡入淡出
(function() {
    var loadingTimeout = null;
    document.body.addEventListener('htmx:beforeRequest', function(ev) {
        var target = ev.detail.target;
        if (!target) return;
        if (target.id === 'gallery-container') {
            var indicator = document.getElementById('gallery-loading-indicator');
            if (indicator) {
                loadingTimeout = setTimeout(function() {
                    indicator.classList.add('gallery-loading-visible');
                }, 150);
            }
            target.classList.add('gallery-swapping');
        }
    });
    document.body.addEventListener('htmx:afterRequest', function(ev) {
        if (loadingTimeout) {
            clearTimeout(loadingTimeout);
            loadingTimeout = null;
        }
        var indicator = document.getElementById('gallery-loading-indicator');
        if (indicator) indicator.classList.remove('gallery-loading-visible');
    });
    document.body.addEventListener('htmx:afterSwap', function(ev) {
        if (ev.detail.target.id === 'gallery-container') {
            ev.detail.target.classList.remove('gallery-swapping');
        }
    });
})();

// HTMX 交换后更新侧栏选中状态、mode 输入、并确保缩略图大小应用；同时将 gallery 响应写入路径缓存
document.body.addEventListener('htmx:afterSwap', function(ev) {
    if (ev.detail.target.id === 'gallery-container') {
        var marker = document.getElementById('current-path-marker');
        const path = marker ? (marker.getAttribute('data-path') || '') : '';

        // 写入 gallery 路径缓存（用于返回已访问路径时直接展示）
        if (marker && window.galleryPathCache) {
            var cacheKey = buildGalleryUrl(path, marker ? {
                mode: marker.getAttribute('data-mode') || 'folder',
                sortBy: marker.getAttribute('data-sort-by') || 'modified_at',
                sortOrder: marker.getAttribute('data-sort-order') || 'desc',
                cols: marker.getAttribute('data-cols') || '4',
                filters: {
                    filter_filename: marker.getAttribute('data-filter-filename') || '',
                    filter_size_min: marker.getAttribute('data-filter-size-min') || '',
                    filter_size_max: marker.getAttribute('data-filter-size-max') || '',
                    filter_date_from: marker.getAttribute('data-filter-date-from') || '',
                    filter_date_to: marker.getAttribute('data-filter-date-to') || '',
                    filter_tag: marker.getAttribute('data-filter-tag') || ''
                }
            } : undefined);
            window.galleryPathCache.set(cacheKey, ev.detail.target.innerHTML);
        }

        // 若为 popstate 恢复，清除标志后不再推入 URL
        if (window._restoringFromPopstate) {
            window._restoringFromPopstate = false;
        } else {
            // 推入或替换 URL，实现浏览器后退逐级返回
            var newUrl = path ? '/?path=' + encodeURIComponent(path) : '/';
            var urlPath = getPathFromUrl();
            if (urlPath === path) {
                history.replaceState({path: path}, '', newUrl);
            } else {
                history.pushState({path: path}, '', newUrl);
            }
        }
        // 将面包屑+操作按钮移到顶栏第一行
        var topBar = ev.detail.target.querySelector('.gallery-top-bar');
        var slot = document.getElementById('gallery-top-slot');
        if (topBar && slot) {
            slot.innerHTML = '';
            slot.appendChild(topBar);
        }

        const saved = localStorage.getItem('fastpic_gallery_cols');
        const sc = document.getElementById('scroll-container');
        if (sc && saved) sc.style.setProperty('--gallery-cols', saved);
        const mode = marker ? (marker.getAttribute('data-mode') || 'folder') : 'folder';
        const sortBy = marker ? (marker.getAttribute('data-sort-by') || 'modified_at') : 'modified_at';
        const sortOrder = marker ? (marker.getAttribute('data-sort-order') || 'desc') : 'desc';
        const pathInput = document.querySelector('[name=path]');
        if (pathInput) pathInput.value = path;
        const modeInput = document.getElementById('mode-input');
        if (modeInput) modeInput.value = mode;
        const sortByInput = document.getElementById('sort-by-input');
        if (sortByInput) sortByInput.value = sortBy;
        const sortOrderInput = document.getElementById('sort-order-input');
        if (sortOrderInput) sortOrderInput.value = sortOrder;

        // 同步筛选状态到全局
        if (marker && typeof window.getFilterState === 'function') {
            var filterState = window.getFilterState();
            // 从 marker 读取实际应用的筛选条件，更新全局状态
            filterState.filter_filename = marker.getAttribute('data-filter-filename') || '';
            filterState.filter_size_min = marker.getAttribute('data-filter-size-min') || '';
            filterState.filter_size_max = marker.getAttribute('data-filter-size-max') || '';
            filterState.filter_date_from = marker.getAttribute('data-filter-date-from') || '';
            filterState.filter_date_to = marker.getAttribute('data-filter-date-to') || '';
            filterState.filter_tag = marker.getAttribute('data-filter-tag') || '';
        }

        // 预取大图模式全量图片列表（5 分钟内不重复请求）
        if (window.folderImagesCache && typeof window.folderImagesCache.prefetch === 'function') {
            window.folderImagesCache.prefetch();
        }
    }
});
