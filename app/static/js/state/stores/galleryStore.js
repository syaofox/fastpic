import { Signal } from '../signals.js';

export const galleryStore = {
    currentPath: new Signal(''),
    viewMode: new Signal('folder'),
    sortBy: new Signal('modified_at'),
    sortOrder: new Signal('desc'),
    filters: new Signal({
        filename: '', sizeMin: '', sizeMax: '',
        dateFrom: '', dateTo: '', tag: '',
    }),
    page: new Signal(1),
    hasNext: new Signal(false),
    _cache: new Map(),
    
    setPath(path) {
        this.currentPath.value = path;
        this.page.value = 1;
        this._cache.clear();
    },
    
    setViewMode(mode) {
        this.viewMode.value = mode;
    },
    
    updateFilters(newFilters) {
        this.filters.value = { ...this.filters.value, ...newFilters };
        this.page.value = 1;
    },
    
    invalidateCache(path = '') {
        if (path) {
            for (const key of this._cache.keys()) {
                if (key.startsWith(path)) this._cache.delete(key);
            }
        } else {
            this._cache.clear();
        }
    },
    
    getCachedData(path, page) {
        const key = `${path}:${page}`;
        return this._cache.get(key);
    },
    
    setCachedData(path, page, data) {
        const key = `${path}:${page}`;
        this._cache.set(key, data);
    },
};
