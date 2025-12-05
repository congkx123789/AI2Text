// API配置
const API_BASE_URL = 'http://localhost:8000';

// DOM元素 - 将在DOMContentLoaded时初始化
let elements = {};
let healthCheckTimer = null;
let isCheckingHealth = false;
let modelsFetchController = null;
let previewObjectUrl = null;

function initializeElements() {
    elements = {
        statusIndicator: document.getElementById('statusIndicator'),
        statusDot: document.querySelector('.status-dot'),
        statusText: document.querySelector('.status-text'),
        refreshModelsBtn: document.getElementById('refreshModelsBtn'),
        checkHealthBtn: document.getElementById('checkHealthBtn'),
        modelList: document.getElementById('modelList'),
        modelSelect: document.getElementById('modelSelect'),
        transcriptionForm: document.getElementById('transcriptionForm'),
        audioFile: document.getElementById('audioFile'),
        fileLabel: document.getElementById('fileLabel'),
        audioPreview: document.getElementById('audioPreview'),
        transcribeBtn: document.getElementById('transcribeBtn'),
        resultSection: document.getElementById('resultSection'),
        resultText: document.getElementById('resultText'),
        confidenceValue: document.getElementById('confidenceValue'),
        processingTime: document.getElementById('processingTime'),
        modelUsed: document.getElementById('modelUsed'),
        copyResultBtn: document.getElementById('copyResultBtn'),
        errorMessage: document.getElementById('errorMessage'),
        beamWidth: document.getElementById('beamWidth'),
        useBeamSearch: document.getElementById('useBeamSearch'),
        useLM: document.getElementById('useLM')
    };
    
    // 验证关键元素是否存在
    const requiredElements = ['transcriptionForm', 'audioFile', 'transcribeBtn', 'modelSelect', 'beamWidth'];
    for (const key of requiredElements) {
        if (!elements[key]) {
            console.error(`缺少必需的DOM元素: ${key}`);
        } else {
            // 确保元素可见且未禁用
            if (elements[key].style) {
                elements[key].style.display = '';
                elements[key].style.visibility = '';
            }
            if (elements[key].disabled !== undefined) {
                elements[key].disabled = false;
            }
        }
    }
    
    // 特别确保beamWidth可见
    if (elements.beamWidth) {
        elements.beamWidth.style.display = 'block';
        elements.beamWidth.style.visibility = 'visible';
        elements.beamWidth.disabled = false;
    }
    
    // 特别确保modelSelect可见且未禁用
    if (elements.modelSelect) {
        elements.modelSelect.style.display = 'block';
        elements.modelSelect.style.visibility = 'visible';
        elements.modelSelect.disabled = false;
    }
}

// 工具函数
function showError(message) {
    elements.errorMessage.textContent = message;
    elements.errorMessage.style.display = 'block';
    setTimeout(() => {
        elements.errorMessage.style.display = 'none';
    }, 5000);
}

function updateStatus(status, text) {
    elements.statusDot.className = 'status-dot ' + status;
    elements.statusText.textContent = text;
}

function setLoading(button, isLoading) {
    const btnText = button.querySelector('.btn-text');
    const btnLoader = button.querySelector('.btn-loader');
    
    if (isLoading) {
        button.disabled = true;
        btnText.style.display = 'none';
        btnLoader.style.display = 'inline-block';
    } else {
        button.disabled = false;
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
    }
}

// API调用函数
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            const data = await response.json();
            updateStatus('connected', '服务运行正常');
            return data;
        } else {
            throw new Error('服务不可用');
        }
    } catch (error) {
        updateStatus('disconnected', '服务连接失败');
        throw error;
    }
}

async function fetchModels() {
    try {
        if (modelsFetchController) {
            modelsFetchController.abort();
        }
        modelsFetchController = new AbortController();

        const response = await fetch(`${API_BASE_URL}/models`, {
            signal: modelsFetchController.signal
        });
        if (!response.ok) throw new Error('获取模型列表失败');
        
        const data = await response.json();
        return data;
    } catch (error) {
        if (error.name === 'AbortError') {
            return { models: [], loaded: [] };
        }
        console.error('获取模型列表错误:', error);
        throw error;
    } finally {
        modelsFetchController = null;
    }
}

function renderModels(models) {
    if (!models || models.length === 0) {
        elements.modelList.innerHTML = '<p class="loading-text">暂无可用模型</p>';
        if (document.getElementById('modelInfo')) {
            document.getElementById('modelInfo').style.display = 'none';
        }
        return;
    }

    // 显示模型信息
    if (document.getElementById('modelInfo')) {
        document.getElementById('modelInfo').style.display = 'block';
        document.getElementById('trainedModelsCount').textContent = models.length;
    }

    elements.modelList.innerHTML = models.map(model => `
        <div class="model-item" data-model="${model.name}">
            <div class="model-name">
                <strong>${model.name}</strong>
                <span class="model-badge">已训练</span>
            </div>
            <div class="model-details">
                <div class="model-size">大小: ${model.size_mb.toFixed(2)} MB</div>
                <div class="model-path">路径: ${model.path}</div>
            </div>
        </div>
    `).join('');

    // 更新选择框
    if (elements.modelSelect) {
        // 清空现有选项
        elements.modelSelect.innerHTML = '';
        
        // 添加默认选项
        const defaultOption = document.createElement('option');
        defaultOption.value = 'default';
        defaultOption.textContent = '默认模型';
        elements.modelSelect.appendChild(defaultOption);
        
        // 添加模型选项
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.name;
            option.textContent = model.name;
            elements.modelSelect.appendChild(option);
        });
    }

    // 添加点击事件
    elements.modelList.querySelectorAll('.model-item').forEach(item => {
        item.addEventListener('click', () => {
            elements.modelList.querySelectorAll('.model-item').forEach(i => i.classList.remove('active'));
            item.classList.add('active');
            elements.modelSelect.value = item.dataset.model;
        });
    });
}

async function loadModels() {
    elements.modelList.innerHTML = '<p class="loading-text">加载中...</p>';
    try {
        const data = await fetchModels();
        if (!data || !Array.isArray(data.models)) {
            return;
        }
        renderModels(data.models);
        
        // 更新已加载模型数量
        if (document.getElementById('loadedModelsCount')) {
            document.getElementById('loadedModelsCount').textContent = Array.isArray(data.loaded) ? data.loaded.length : 0;
        }
    } catch (error) {
        elements.modelList.innerHTML = '<p class="loading-text" style="color: var(--error-color);">加载失败: ' + error.message + '</p>';
        if (document.getElementById('modelInfo')) {
            document.getElementById('modelInfo').style.display = 'none';
        }
    }
}

async function transcribeAudio(formData) {
    try {
        const response = await fetch(`${API_BASE_URL}/transcribe`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: '转录失败' }));
            throw new Error(errorData.detail || '转录失败');
        }

        return await response.json();
    } catch (error) {
        console.error('转录错误:', error);
        throw error;
    }
}

function displayResult(result) {
    const text = result.text || '';
    
    // 如果文本为空或只有空白，显示提示信息
    if (!text || text.trim() === '') {
        elements.resultText.innerHTML = '<div style="color: var(--warning-color); padding: 20px; text-align: center;">' +
            '<strong>⚠️ 转录结果为空</strong><br><br>' +
            '可能的原因：<br>' +
            '1. 模型训练不充分<br>' +
            '2. 音频内容不在训练数据中<br>' +
            '3. 模型需要更多训练数据<br><br>' +
            '置信度: ' + (result.confidence * 100).toFixed(2) + '% (较低)<br>' +
            '建议：检查后端窗口中的调试信息，或使用更多训练数据重新训练模型' +
            '</div>';
    } else {
        elements.resultText.textContent = text;
    }
    
    elements.confidenceValue.textContent = (result.confidence * 100).toFixed(2) + '%';
    elements.processingTime.textContent = result.processing_time.toFixed(2) + ' 秒';
    elements.modelUsed.textContent = result.model_name;
    elements.resultSection.style.display = 'block';
    
    // 滚动到结果区域
    elements.resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// 注册事件监听器
function setupEventListeners() {
    if (elements.refreshModelsBtn) {
        elements.refreshModelsBtn.addEventListener('click', loadModels);
    }
    
    if (elements.checkHealthBtn) {
        elements.checkHealthBtn.addEventListener('click', async () => {
            try {
                const health = await checkHealth();
                alert('服务状态: 正常\n已加载模型: ' + health.models_loaded);
            } catch (error) {
                showError('服务检查失败: ' + error.message);
            }
        });
    }

    if (elements.audioFile) {
        elements.audioFile.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                if (elements.fileLabel) {
                    elements.fileLabel.textContent = file.name + ' (' + (file.size / 1024 / 1024).toFixed(2) + ' MB)';
                    elements.fileLabel.parentElement.classList.add('has-file');
                }
                
                // 显示音频预览
                if (elements.audioPreview) {
                    if (previewObjectUrl) {
                        URL.revokeObjectURL(previewObjectUrl);
                    }
                    previewObjectUrl = URL.createObjectURL(file);
                    elements.audioPreview.src = previewObjectUrl;
                    elements.audioPreview.style.display = 'block';
                }
            } else {
                if (elements.fileLabel) {
                    elements.fileLabel.textContent = '点击选择或拖拽文件到此处';
                    elements.fileLabel.parentElement.classList.remove('has-file');
                }
                if (elements.audioPreview) {
                    if (previewObjectUrl) {
                        URL.revokeObjectURL(previewObjectUrl);
                        previewObjectUrl = null;
                    }
                    elements.audioPreview.style.display = 'none';
                }
            }
        });
    }

    // 拖拽上传
    if (elements.audioFile && elements.audioFile.parentElement) {
        const fileInputWrapper = elements.audioFile.parentElement;
        fileInputWrapper.addEventListener('dragover', (e) => {
            e.preventDefault();
            fileInputWrapper.style.borderColor = 'var(--primary-color)';
            fileInputWrapper.style.background = 'rgba(99, 102, 241, 0.1)';
        });

        fileInputWrapper.addEventListener('dragleave', (e) => {
            e.preventDefault();
            fileInputWrapper.style.borderColor = 'var(--border-color)';
            fileInputWrapper.style.background = 'var(--bg-color)';
        });

        fileInputWrapper.addEventListener('drop', (e) => {
            e.preventDefault();
            fileInputWrapper.style.borderColor = 'var(--border-color)';
            fileInputWrapper.style.background = 'var(--bg-color)';
            
            const files = e.dataTransfer.files;
            if (files.length > 0 && elements.audioFile) {
                elements.audioFile.files = files;
                elements.audioFile.dispatchEvent(new Event('change'));
            }
        });
    }

    if (elements.transcriptionForm) {
        elements.transcriptionForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const file = elements.audioFile?.files?.[0];
            if (!file) {
                showError('请先选择音频文件');
                return;
            }

            setLoading(elements.transcribeBtn, true);
            if (elements.resultSection) {
                elements.resultSection.style.display = 'none';
            }

            try {
                const formData = new FormData();
                formData.append('audio', file);
                formData.append('model_name', elements.modelSelect?.value || 'default');
                formData.append('use_beam_search', elements.useBeamSearch ? elements.useBeamSearch.checked : true);
                formData.append('beam_width', elements.beamWidth ? elements.beamWidth.value : '5');
                formData.append('use_lm', elements.useLM ? elements.useLM.checked : false);

                const result = await transcribeAudio(formData);
                displayResult(result);
            } catch (error) {
                showError('转录失败: ' + error.message);
            } finally {
                setLoading(elements.transcribeBtn, false);
            }
        });
    }

    if (elements.copyResultBtn) {
        elements.copyResultBtn.addEventListener('click', () => {
            const text = elements.resultText?.textContent || '';
            navigator.clipboard.writeText(text).then(() => {
                const originalText = elements.copyResultBtn.textContent;
                elements.copyResultBtn.textContent = '已复制!';
                setTimeout(() => {
                    elements.copyResultBtn.textContent = originalText;
                }, 2000);
            }).catch(() => {
                showError('复制失败');
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
            // 静默失败
        } finally {
            isCheckingHealth = false;
            healthCheckTimer = setTimeout(tick, interval);
        }
    };

    healthCheckTimer = setTimeout(tick, interval);
}

// 初始化
async function init() {
    // 初始化DOM元素
    initializeElements();
    
    // 注册事件监听器
    setupEventListeners();
    
    // 检查服务状态
    try {
        await checkHealth();
    } catch (error) {
        showError('无法连接到API服务，请确保后端服务正在运行在 ' + API_BASE_URL);
    }

    // 加载模型列表
    await loadModels();

    // 定期检查服务状态
    startHealthMonitoring();
}

// 页面加载完成后初始化
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    // DOM已经加载完成
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

