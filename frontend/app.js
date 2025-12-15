// API Configuration
let API_BASE_URL = 'http://localhost:8000';
const AI2TEXT_TRANSFORMER_URL = 'http://localhost:8000'; // /transcribe (field: audio)
const WHISPER_CT2_URL = 'http://localhost:8002';         // /transcribe/upload (field: file)
const CTC_API_URL = 'http://localhost:8001';             // /transcribe (field: file)
const WHISPER_BASE_URL = 'http://localhost:8003';        // /transcribe/upload (field: file)
let currentAPI = 'server';

// DOM Elements
let elements = {};
let healthCheckTimer = null;
let isCheckingHealth = false;
let previewObjectUrl = null;
let modelsFetchController = null;

function initializeElements() {
    elements = {
        statusIndicator: document.getElementById('statusIndicator'),
        statusDot: document.querySelector('.status-dot'),
        statusText: document.querySelector('.status-text'),
        apiSelect: document.getElementById('apiSelect'),
        modelSection: document.getElementById('modelSection'),
        refreshModelsBtn: document.getElementById('refreshModelsBtn'),
        checkHealthBtn: document.getElementById('checkHealthBtn'),
        modelList: document.getElementById('modelList'),
        modelInfo: document.getElementById('modelInfo'),
        trainedModelsCount: document.getElementById('trainedModelsCount'),
        loadedModelsCount: document.getElementById('loadedModelsCount'),
        modelSelect: document.getElementById('modelSelect'),
        modelSelectionGroup: document.getElementById('modelSelectionGroup'),
        backendSelectionGroup: document.getElementById('backendSelectionGroup'),
        beamSearchOptions: document.getElementById('beamSearchOptions'),
        conversionForm: document.getElementById('conversionForm'),
        modeSelect: document.getElementById('modeSelect'),
        fileInput: document.getElementById('fileInput'),
        fileInputWrapper: document.getElementById('fileInputWrapper'),
        fileLabel: document.getElementById('fileLabel'),
        audioPreview: document.getElementById('audioPreview'),
        imagePreview: document.getElementById('imagePreview'),
        audioOptions: document.getElementById('audioOptions'),
        imageOptions: document.getElementById('imageOptions'),
        sttBackend: document.getElementById('sttBackend'),
        ocrBackend: document.getElementById('ocrBackend'),
        audioLanguage: document.getElementById('audioLanguage'),
        imageLanguage: document.getElementById('imageLanguage'),
        beamWidth: document.getElementById('beamWidth'),
        useBeamSearch: document.getElementById('useBeamSearch'),
        useLM: document.getElementById('useLM'),
        noiseReduction: document.getElementById('noiseReduction'),
        enhanceSpeech: document.getElementById('enhanceSpeech'),
        normalize: document.getElementById('normalize'),
        removeHum: document.getElementById('removeHum'),
        highPass: document.getElementById('highPass'),
        lowPass: document.getElementById('lowPass'),
        convertBtn: document.getElementById('convertBtn'),
        resultSection: document.getElementById('resultSection'),
        resultText: document.getElementById('resultText'),
        resultMetrics: document.getElementById('resultMetrics'),
        backendValue: document.getElementById('backendValue'),
        languageValue: document.getElementById('languageValue'),
        confidenceValue: document.getElementById('confidenceValue'),
        processingTimeValue: document.getElementById('processingTimeValue'),
        modelUsedValue: document.getElementById('modelUsedValue'),
        confidenceMetric: document.getElementById('confidenceMetric'),
        processingTimeMetric: document.getElementById('processingTimeMetric'),
        modelUsedMetric: document.getElementById('modelUsedMetric'),
        copyResultBtn: document.getElementById('copyResultBtn'),
        errorMessage: document.getElementById('errorMessage')
    };
}

// Utility Functions
function showError(message) {
    if (elements.errorMessage) {
        elements.errorMessage.textContent = message;
        elements.errorMessage.style.display = 'block';
        setTimeout(() => {
            elements.errorMessage.style.display = 'none';
        }, 5000);
    }
}

function updateStatus(status, text) {
    if (elements.statusDot && elements.statusText) {
        elements.statusDot.className = 'status-dot ' + status;
        elements.statusText.textContent = text;
    }
}

function setLoading(button, isLoading) {
    if (!button) return;
    const btnText = button.querySelector('.btn-text');
    const btnLoader = button.querySelector('.btn-loader');
    
    if (isLoading) {
        button.disabled = true;
        if (btnText) btnText.style.display = 'none';
        if (btnLoader) btnLoader.style.display = 'inline-block';
    } else {
        button.disabled = false;
        if (btnText) btnText.style.display = 'inline';
        if (btnLoader) btnLoader.style.display = 'none';
    }
}

// API Functions
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            const data = await response.json();
            updateStatus('connected', 'Service is running');
            return data;
        } else {
            throw new Error('Service unavailable');
        }
    } catch (error) {
        updateStatus('disconnected', 'Service connection failed');
        throw error;
    }
}

// Model Management (for AI2Text API)
async function fetchModels() {
    // Only call when backend supports /models (ai2text-transformer legacy list, whisper-ct2 currently no-op)
    if (currentAPI !== 'ai2text-transformer') {
        return { models: [], loaded: [] };
    }
    try {
        if (modelsFetchController) {
            modelsFetchController.abort();
        }
        modelsFetchController = new AbortController();

        const response = await fetch(`${API_BASE_URL}/models`, {
            signal: modelsFetchController.signal
        });
        if (!response.ok) throw new Error('Failed to fetch model list');
        
        const data = await response.json();
        return data;
    } catch (error) {
        if (error.name === 'AbortError') {
            return { models: [], loaded: [] };
        }
        console.error('Error fetching models:', error);
        throw error;
    } finally {
        modelsFetchController = null;
    }
}

function renderModels(models) {
    if (!models || models.length === 0) {
        if (elements.modelList) {
            elements.modelList.innerHTML = '<p class="loading-text">No models available</p>';
        }
        if (elements.modelInfo) {
            elements.modelInfo.style.display = 'none';
        }
        return;
    }

    // Show model info
    if (elements.modelInfo) {
        elements.modelInfo.style.display = 'block';
        if (elements.trainedModelsCount) {
            elements.trainedModelsCount.textContent = models.length;
        }
    }

    // Render model list
    if (elements.modelList) {
        elements.modelList.innerHTML = models.map(model => `
            <div class="model-item" data-model="${model.name}">
                <div class="model-name">
                    <strong>${model.name}</strong>
                    <span class="model-badge">Trained</span>
                </div>
                <div class="model-details">
                    <div class="model-size">Size: ${model.size_mb ? model.size_mb.toFixed(2) : 'N/A'} MB</div>
                    <div class="model-path">Path: ${model.path}</div>
                </div>
            </div>
        `).join('');
    }

    // Update model select dropdown
    if (elements.modelSelect) {
        elements.modelSelect.innerHTML = '';
        const defaultOption = document.createElement('option');
        defaultOption.value = 'default';
        defaultOption.textContent = 'Default Model';
        elements.modelSelect.appendChild(defaultOption);
        
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.name;
            option.textContent = model.name;
            elements.modelSelect.appendChild(option);
        });
    }

    // Add click events
    if (elements.modelList) {
        elements.modelList.querySelectorAll('.model-item').forEach(item => {
            item.addEventListener('click', () => {
                elements.modelList.querySelectorAll('.model-item').forEach(i => i.classList.remove('active'));
                item.classList.add('active');
                if (elements.modelSelect) {
                    elements.modelSelect.value = item.dataset.model;
                }
            });
        });
    }
}

async function loadModels() {
    if (elements.modelList) {
        elements.modelList.innerHTML = '<p class="loading-text">Loading...</p>';
    }
    try {
        const data = await fetchModels();
        if (!data || !Array.isArray(data.models)) {
            return;
        }
        renderModels(data.models);
        
        // Update loaded models count
        if (elements.loadedModelsCount) {
            elements.loadedModelsCount.textContent = Array.isArray(data.loaded) ? data.loaded.length : 0;
        }
    } catch (error) {
        if (elements.modelList) {
            elements.modelList.innerHTML = '<p class="loading-text" style="color: var(--error-color);">Failed to load: ' + error.message + '</p>';
        }
        if (elements.modelInfo) {
            elements.modelInfo.style.display = 'none';
        }
    }
}

// Detect File Type
function detectFileType(file) {
    if (!file) return null;
    const ext = file.name.split('.').pop().toLowerCase();
    const audioExts = ['wav', 'mp3', 'm4a', 'flac', 'ogg', 'aac', 'wma', 'webm', 'mp4'];
    const imageExts = ['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff', 'webp', 'pbm', 'ppm'];
    
    if (audioExts.includes(ext)) return 'audio';
    if (imageExts.includes(ext)) return 'image';
    return null;
}

// Update UI Based on API Selection
function updateAPISelection() {
    const api = elements.apiSelect?.value || 'server';
    currentAPI = api;
    
    // Default hide sections
    if (elements.modelSection) elements.modelSection.style.display = 'none';
    if (elements.modelSelectionGroup) elements.modelSelectionGroup.style.display = 'none';
    if (elements.backendSelectionGroup) elements.backendSelectionGroup.style.display = 'block';
    if (elements.beamSearchOptions) elements.beamSearchOptions.style.display = 'none';
    
    if (api === 'ai2text-transformer') {
        API_BASE_URL = AI2TEXT_TRANSFORMER_URL;
        // Transformer: no model list UI
        if (elements.backendSelectionGroup) elements.backendSelectionGroup.style.display = 'none';
    } else if (api === 'whisper-ct2') {
        API_BASE_URL = WHISPER_CT2_URL;
        if (elements.modelSection) elements.modelSection.style.display = 'block';
        if (elements.modelSelectionGroup) elements.modelSelectionGroup.style.display = 'block';
        if (elements.backendSelectionGroup) elements.backendSelectionGroup.style.display = 'none';
        if (elements.beamSearchOptions) elements.beamSearchOptions.style.display = 'block';
        loadModels(); // reuse model list UI for CT2
    } else if (api === 'whisper-ctc') {
        API_BASE_URL = CTC_API_URL;
        // CTC: no model list, no beam options
    } else if (api === 'whisper-base') {
        API_BASE_URL = WHISPER_BASE_URL;
        // Base Whisper: use /transcribe
    } else {
        API_BASE_URL = 'http://localhost:8000'; // legacy server api /api/convert
    }
    
    // Recheck health
    checkHealth().catch(() => {});
}

// Update UI Based on Mode Selection
function updateModeUI() {
    const mode = elements.modeSelect?.value || 'audio';
    const file = elements.fileInput?.files?.[0];
    
    // Hide all options
    if (elements.audioOptions) elements.audioOptions.style.display = 'none';
    
    // Audio-only: always show audio options
    if (elements.audioOptions) {
        elements.audioOptions.style.display = 'block';
    }
}

// Convert File (Server API - legacy, audio only here)
async function convertFile(formData) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/convert`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Conversion failed' }));
            throw new Error(errorData.detail || 'Conversion failed');
        }

        return await response.json();
    } catch (error) {
        console.error('Conversion error:', error);
        throw error;
    }
}

// Transcribe Audio via /transcribe (generic)
async function transcribeServer(formData) {
    try {
        const response = await fetch(`${API_BASE_URL}/transcribe`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Transcription failed' }));
            throw new Error(errorData.detail || 'Transcription failed');
        }

        return await response.json();
    } catch (error) {
        console.error('Transcription error (server):', error);
        throw error;
    }
}

// Transcribe Audio via /transcribe/upload (Whisper-style upload)
async function transcribeUpload(formData) {
    try {
        const response = await fetch(`${API_BASE_URL}/transcribe/upload`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Transcription failed' }));
            throw new Error(errorData.detail || 'Transcription failed');
        }

        return await response.json();
    } catch (error) {
        console.error('Transcription error:', error);
        throw error;
    }
}

// Display Result
function displayResult(result) {
    const text = result.text || '';
    
    if (!text || text.trim() === '') {
        if (elements.resultText) {
            elements.resultText.innerHTML = '<div style="color: var(--warning-color); padding: 20px; text-align: center;">' +
                '<strong>⚠️ Conversion result is empty</strong><br><br>' +
                'Possible reasons:<br>' +
                '1. File content cannot be recognized<br>' +
                '2. Language setting is incorrect<br>' +
                '3. Backend service configuration issue' +
                '</div>';
        }
    } else {
        if (elements.resultText) {
            elements.resultText.textContent = text;
        }
    }
    
    // Display metadata
    if (result.meta) {
        if (result.meta.backend && elements.backendValue) {
            elements.backendValue.textContent = result.meta.backend;
        }
        if (result.meta.language && elements.languageValue) {
            elements.languageValue.textContent = result.meta.language;
        }
    }
    
    // Display AI2Text API specific metrics
    if (result.confidence !== undefined) {
        if (elements.confidenceValue) {
            elements.confidenceValue.textContent = (result.confidence * 100).toFixed(2) + '%';
        }
        if (elements.confidenceMetric) {
            elements.confidenceMetric.style.display = 'flex';
        }
    }
    
    if (result.processing_time !== undefined) {
        if (elements.processingTimeValue) {
            elements.processingTimeValue.textContent = result.processing_time.toFixed(2) + 's';
        }
        if (elements.processingTimeMetric) {
            elements.processingTimeMetric.style.display = 'flex';
        }
    }
    
    if (result.model_name) {
        if (elements.modelUsedValue) {
            elements.modelUsedValue.textContent = result.model_name;
        }
        if (elements.modelUsedMetric) {
            elements.modelUsedMetric.style.display = 'flex';
        }
    }
    
    if (elements.resultSection) {
        elements.resultSection.style.display = 'block';
        elements.resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
}

// Register Event Listeners
function setupEventListeners() {
    // API selection change
    if (elements.apiSelect) {
        elements.apiSelect.addEventListener('change', updateAPISelection);
    }
    
    // Model management
    if (elements.refreshModelsBtn) {
        elements.refreshModelsBtn.addEventListener('click', loadModels);
    }
    
    if (elements.checkHealthBtn) {
        elements.checkHealthBtn.addEventListener('click', async () => {
            try {
                const health = await checkHealth();
                alert('Service Status: OK\nLoaded Models: ' + (health.models_loaded ? health.models_loaded.length : 0));
            } catch (error) {
                showError('Health check failed: ' + error.message);
            }
        });
    }

    // Mode selection change
    if (elements.modeSelect) {
        elements.modeSelect.addEventListener('change', updateModeUI);
    }

    // File input change
    if (elements.fileInput) {
        elements.fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                if (elements.fileLabel) {
                    elements.fileLabel.textContent = file.name + ' (' + (file.size / 1024 / 1024).toFixed(2) + ' MB)';
                    elements.fileLabel.parentElement.classList.add('has-file');
                }
                
                const fileType = detectFileType(file);
                
                if (elements.audioPreview) elements.audioPreview.style.display = 'none';
                if (elements.imagePreview) elements.imagePreview.style.display = 'none';
                
                if (fileType === 'audio' && elements.audioPreview) {
                    if (previewObjectUrl) {
                        URL.revokeObjectURL(previewObjectUrl);
                    }
                    previewObjectUrl = URL.createObjectURL(file);
                    elements.audioPreview.src = previewObjectUrl;
                    elements.audioPreview.style.display = 'block';
                } else if (fileType === 'image' && elements.imagePreview) {
                    if (previewObjectUrl) {
                        URL.revokeObjectURL(previewObjectUrl);
                    }
                    previewObjectUrl = URL.createObjectURL(file);
                    elements.imagePreview.src = previewObjectUrl;
                    elements.imagePreview.style.display = 'block';
                }
                
                updateModeUI();
            } else {
                if (elements.fileLabel) {
                    elements.fileLabel.textContent = 'Click to select or drag and drop file here';
                    elements.fileLabel.parentElement.classList.remove('has-file');
                }
                if (elements.audioPreview) elements.audioPreview.style.display = 'none';
                if (elements.imagePreview) elements.imagePreview.style.display = 'none';
                if (previewObjectUrl) {
                    URL.revokeObjectURL(previewObjectUrl);
                    previewObjectUrl = null;
                }
            }
        });
    }

    // Drag and drop
    if (elements.fileInputWrapper) {
        elements.fileInputWrapper.addEventListener('dragover', (e) => {
            e.preventDefault();
            elements.fileInputWrapper.style.borderColor = 'var(--primary-color)';
            elements.fileInputWrapper.style.background = 'rgba(99, 102, 241, 0.1)';
        });

        elements.fileInputWrapper.addEventListener('dragleave', (e) => {
            e.preventDefault();
            elements.fileInputWrapper.style.borderColor = 'var(--border-color)';
            elements.fileInputWrapper.style.background = 'var(--bg-color)';
        });

        elements.fileInputWrapper.addEventListener('drop', (e) => {
            e.preventDefault();
            elements.fileInputWrapper.style.borderColor = 'var(--border-color)';
            elements.fileInputWrapper.style.background = 'var(--bg-color)';
            
            const files = e.dataTransfer.files;
            if (files.length > 0 && elements.fileInput) {
                elements.fileInput.files = files;
                elements.fileInput.dispatchEvent(new Event('change'));
            }
        });
    }

    // Form submit
    if (elements.conversionForm) {
        elements.conversionForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const file = elements.fileInput?.files?.[0];
            if (!file) {
                showError('Please select a file first');
                return;
            }

            setLoading(elements.convertBtn, true);
            if (elements.resultSection) {
                elements.resultSection.style.display = 'none';
            }
            
            // Hide metrics initially
            if (elements.confidenceMetric) elements.confidenceMetric.style.display = 'none';
            if (elements.processingTimeMetric) elements.processingTimeMetric.style.display = 'none';
            if (elements.modelUsedMetric) elements.modelUsedMetric.style.display = 'none';

            try {
                const mode = elements.modeSelect?.value || 'audio';
                const fileType = 'audio';
                
                // Only audio supported in this UI
                if (fileType !== 'audio' && mode !== 'audio') {
                    showError('Hiện chỉ hỗ trợ audio. Vui lòng chọn file audio.');
                    return;
                }

                // Whisper CT2 (8002) with model select -> /transcribe/upload, field "file"
                if (currentAPI === 'whisper-ct2') {
                    const formData = new FormData();
                    formData.append('file', file);
                    formData.append('model_name', elements.modelSelect?.value || 'default');
                    formData.append('use_beam_search', elements.useBeamSearch?.checked || false);
                    formData.append('beam_width', elements.beamWidth?.value || '5');
                    formData.append('use_lm', elements.useLM?.checked || false);
                    
                    const result = await transcribeUpload(formData);
                    displayResult(result);
                    return;
                }

                const audioLanguage = elements.audioLanguage?.value || '';

                // Whisper Base (8003) -> /transcribe/upload, field "file"
                if (currentAPI === 'whisper-base') {
                    const formData = new FormData();
                    formData.append('file', file);
                    if (audioLanguage) formData.append('language', audioLanguage);
                    const result = await transcribeUpload(formData);
                    displayResult(result);
                    return;
                }

                // CTC (8001) -> /transcribe, field "file"
                if (currentAPI === 'whisper-ctc') {
                    const formData = new FormData();
                    formData.append('file', file);
                    if (audioLanguage) formData.append('language', audioLanguage);
                    const result = await transcribeServer(formData);
                    displayResult(result);
                    return;
                }

                // AI2Text Transformer (8000 /transcribe, field "audio")
                if (currentAPI === 'ai2text-transformer') {
                    const formData = new FormData();
                    formData.append('audio', file);
                    if (audioLanguage) formData.append('language', audioLanguage);
                    const result = await transcribeServer(formData);
                    displayResult(result);
                    return;
                }

                // Legacy server (8000 /api/convert) fallback
                if (currentAPI === 'server') {
                    try {
                        const fallback = new FormData();
                        fallback.append('file', file);
                        fallback.append('mode', 'audio');
                        const result = await convertFile(fallback);
                        displayResult(result);
                        return;
                    } catch (err2) {
                        showError('Conversion failed: ' + (err2.message));
                        return;
                    }
                }
            } catch (error) {
                showError('Conversion failed: ' + error.message);
            } finally {
                setLoading(elements.convertBtn, false);
            }
        });
    }

    // Copy result
    if (elements.copyResultBtn) {
        elements.copyResultBtn.addEventListener('click', () => {
            const text = elements.resultText?.textContent || '';
            navigator.clipboard.writeText(text).then(() => {
                const originalText = elements.copyResultBtn.textContent;
                elements.copyResultBtn.textContent = 'Copied!';
                setTimeout(() => {
                    elements.copyResultBtn.textContent = originalText;
                }, 2000);
            }).catch(() => {
                showError('Copy failed');
            });
        });
    }
}

function startHealthMonitoring(interval = 30000) {
    if (healthCheckTimer) {
        clearTimeout(healthCheckTimer);
    }

    const tick = async () => {
        if (isCheckingHealth) {
            healthCheckTimer = setTimeout(tick, interval);
            return;
        }

        try {
            isCheckingHealth = true;
            await checkHealth();
        } catch (error) {
            // Silent fail
        } finally {
            isCheckingHealth = false;
            healthCheckTimer = setTimeout(tick, interval);
        }
    };

    healthCheckTimer = setTimeout(tick, interval);
}

// Initialize
async function init() {
    initializeElements();
    setupEventListeners();
    updateAPISelection();
    
    try {
        await checkHealth();
    } catch (error) {
        showError('Unable to connect to API service. Please ensure the backend service is running.');
    }

    startHealthMonitoring();
}

// Initialize when page loads
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

window.addEventListener('beforeunload', () => {
    if (previewObjectUrl) {
        URL.revokeObjectURL(previewObjectUrl);
        previewObjectUrl = null;
    }
    if (healthCheckTimer) {
        clearTimeout(healthCheckTimer);
    }
    if (modelsFetchController) {
        modelsFetchController.abort();
        modelsFetchController = null;
    }
});
