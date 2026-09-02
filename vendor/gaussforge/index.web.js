class GaussForgeBase {
    constructor() {
        this.module = null;
        this.instance = null;
        this.initPromise = null;
    }
    async init(moduleFactory) {
        if (this.initPromise)
            return this.initPromise;
        this.initPromise = (async () => {
            try {
                let moduleInstance;
                if (moduleFactory) {
                    moduleInstance = moduleFactory();
                }
                else {
                    const createModule = await this.importWasmModule();
                    const factory = createModule.default;
                    moduleInstance = typeof factory === 'function' ? factory() : factory;
                }
                if (moduleInstance && typeof moduleInstance.then === 'function') {
                    moduleInstance = await moduleInstance;
                }
                this.module = moduleInstance;
                this.instance = new this.module.GaussForgeWASM();
            }
            catch (error) {
                this.initPromise = null;
                throw new Error(`GaussForge Init Failed: ${error instanceof Error ? error.message : String(error)}`);
            }
        })();
        return this.initPromise;
    }
    ensureInitialized() {
        if (!this.instance)
            throw new Error('GaussForge not initialized. Call init() first.');
    }
    async read(data, format, options = {}) {
        this.ensureInitialized();
        const input = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
        const result = this.instance.read(input, format, options.strict || false);
        if (result.error)
            throw new Error(result.error);
        return {
            data: result.data,
            ...(result.warning && { warning: result.warning })
        };
    }
    async write(ir, format, options = {}) {
        this.ensureInitialized();
        const result = this.instance.write(ir, format, options.strict || false, options.spzVersion ?? 3);
        if (result.error)
            throw new Error(result.error);
        return {
            data: result.data
        };
    }
    async convert(data, inFmt, outFmt, options = {}) {
        this.ensureInitialized();
        const input = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
        const result = this.instance.convert(input, inFmt, outFmt, options.strict || false, options.includeInfo || false, options.spzVersion ?? 3);
        if (result.error)
            throw new Error(result.error);
        return {
            data: result.data,
            ...(result.modelInfo && { modelInfo: result.modelInfo })
        };
    }
    getSupportedFormats() {
        this.ensureInitialized();
        return this.instance.getSupportedFormats();
    }
    getVersion() {
        this.ensureInitialized();
        return this.instance.getVersion();
    }
    async getModelInfo(data, format, options = {}) {
        this.ensureInitialized();
        const input = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
        const result = this.instance.getModelInfo(input, format, options.fileSize || 0);
        if (result.error)
            return { error: result.error };
        return {
            data: result.data
        };
    }
    /**
     * Check if a format is supported
     * @param format - Format name to check
     * @returns true if the format is supported, false otherwise
     */
    isFormatSupported(format) {
        this.ensureInitialized();
        const formats = this.getSupportedFormats();
        return formats.includes(format);
    }
    /**
     * Explicitly destroy C++ instance to prevent memory leaks
     */
    dispose() {
        if (this.instance) {
            this.instance.delete();
            this.instance = null;
        }
        this.module = null;
        this.initPromise = null;
    }
}

class GaussForge extends GaussForgeBase {
    async importWasmModule() {
        // @ts-ignore
        return import('./gauss_forge.web.js');
    }
}
let _instance = null;
async function createGaussForge(factory) {
    if (!_instance) {
        _instance = new GaussForge();
        await _instance.init(factory);
    }
    return _instance;
}
function destroyGaussForge() {
    if (_instance) {
        _instance.dispose();
        _instance = null;
    }
}

export { GaussForge, createGaussForge, destroyGaussForge };
//# sourceMappingURL=index.web.js.map
