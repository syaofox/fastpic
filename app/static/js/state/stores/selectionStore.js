import { Signal } from '../signals.js';

export const selectionStore = {
    _selectedImages: new Set(),
    _selectedFolders: new Set(),
    _selectMode: new Signal(false),
    
    get selectedImages() { return this._selectedImages; },
    get selectedFolders() { return this._selectedFolders; },
    get selectMode() { return this._selectMode.value; },
    
    toggleImage(id) {
        if (this._selectedImages.has(id)) {
            this._selectedImages.delete(id);
        } else {
            this._selectedImages.add(id);
        }
    },
    
    toggleFolder(path) {
        if (this._selectedFolders.has(path)) {
            this._selectedFolders.delete(path);
        } else {
            this._selectedFolders.add(path);
        }
    },
    
    clearSelection() {
        this._selectedImages.clear();
        this._selectedFolders.clear();
        this._selectMode.value = false;
    },
    
    setSelectMode(enabled) { this._selectMode.value = enabled; },
};
