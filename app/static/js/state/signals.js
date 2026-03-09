class Signal {
    constructor(initialValue) {
        this._value = initialValue;
        this._callbacks = new Set();
    }
    
    get value() {
        return this._value;
    }
    
    set value(newValue) {
        if (this._value !== newValue) {
            this._value = newValue;
            this._notify();
        }
    }
    
    subscribe(callback) {
        this._callbacks.add(callback);
        return () => this._callbacks.delete(callback);
    }
    
    _notify() {
        this._callbacks.forEach(cb => cb(this._value));
    }
}

function computed(computeFn) {
    const signal = new Signal(computeFn());
    let lastValue = signal.value;
    
    const tracker = new Signal(0);
    tracker.subscribe(() => {
        const newValue = computeFn();
        if (newValue !== lastValue) {
            lastValue = newValue;
            signal._value = newValue;
            signal._notify();
        }
    });
    
    return {
        get value() { return signal.value; },
        subscribe: signal.subscribe.bind(signal),
    };
}

export { Signal, computed };
