/**
 * AgentASR Frontend Application - 现代化设计版
 * 负责前端交互逻辑：录音、API调用、SSE流式响应处理
 */

// ==================== 全局状态 ====================
const state = {
    isRecording: false,
    isProcessing: false,
    mediaRecorder: null,
    audioChunks: [],
    currentUser: 'Guest',
    speakers: []
};

// ==================== DOM 元素 ====================
const elements = {
    statusIndicator: document.getElementById('statusIndicator'),
    statusDot: document.querySelector('.status-dot'),
    statusText: document.querySelector('.status-text'),
    userId: document.getElementById('userId'),
    refreshUsers: document.getElementById('refreshUsers'),
    chatMessages: document.getElementById('chatMessages'),
    messageInput: document.getElementById('messageInput'),
    recordBtn: document.getElementById('recordBtn'),
    sendBtn: document.getElementById('sendBtn'),
    registerBtn: document.getElementById('registerBtn'),
    recordingOverlay: document.getElementById('recordingOverlay'),
    stopRecordBtn: document.getElementById('stopRecordBtn'),
    audioPlayer: document.getElementById('audioPlayer')
};

// ==================== 初始化 ====================
document.addEventListener('DOMContentLoaded', async () => {
    console.log('🚀 AgentASR Frontend 初始化中...');

    // 检查服务器状态
    await checkServerHealth();

    // 加载用户列表
    await loadSpeakers();

    // 绑定事件监听器
    bindEvents();

    // 显示欢迎动画
    setTimeout(() => {
        const welcome = document.querySelector('.message.system');
        if (welcome) {
            welcome.style.opacity = '1';
        }
    }, 300);

    console.log('✅ 初始化完成');
});

// ==================== 事件绑定 ====================
function bindEvents() {
    // 发送消息
    elements.sendBtn.addEventListener('click', handleSendMessage);
    elements.messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleSendMessage();
    });

    // 录音控制
    elements.recordBtn.addEventListener('click', startRecording);
    elements.stopRecordBtn.addEventListener('click', stopRecording);

    // 刷新用户列表
    elements.refreshUsers.addEventListener('click', loadSpeakers);
    elements.userId.addEventListener('change', (e) => {
        state.currentUser = e.target.value;
        console.log('用户切换到:', state.currentUser);
        addSystemMessage(`已切换到用户: ${state.currentUser}`);
    });

    // 声纹注册
    elements.registerBtn.addEventListener('click', handleRegisterSpeaker);
}

// ==================== API 调用 ====================
async function apiRequest(endpoint, method = 'GET', data = null, isFormData = false) {
    const headers = {};
    if (!isFormData) {
        headers['Content-Type'] = 'application/json';
    }

    const options = {
        method,
        headers,
    };

    if (data) {
        if (isFormData) {
            options.body = data;
        } else {
            options.body = JSON.stringify(data);
        }
    }

    try {
        const response = await fetch(endpoint, options);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        return await response.json();
    } catch (error) {
        console.error('API 请求失败:', error);
        throw error;
    }
}

async function checkServerHealth() {
    try {
        const health = await apiRequest('/api/health');
        updateStatus('connected');
        console.log('服务器状态:', health);
        return health;
    } catch (error) {
        updateStatus('disconnected');
        addSystemMessage('⚠️ 无法连接到服务器，请检查服务是否启动', 'error');
        throw error;
    }
}

async function loadSpeakers() {
    try {
        const result = await apiRequest('/api/speaker/list');
        state.speakers = result.speakers;

        // 更新用户下拉框
        elements.userId.innerHTML = '<option value="Guest">👤 访客</option>';
        result.speakers.forEach(speaker => {
            const option = document.createElement('option');
            option.value = speaker;
            option.textContent = speaker;
            elements.userId.appendChild(option);
        });

        console.log('加载用户列表:', state.speakers);
    } catch (error) {
        console.error('加载用户列表失败:', error);
    }
}

// ==================== 消息处理 ====================
async function handleSendMessage() {
    const message = elements.messageInput.value.trim();
    if (!message || state.isProcessing) return;

    elements.messageInput.value = '';
    await sendMessage(message);
}

async function sendMessage(message, audioBlob = null) {
    state.isProcessing = true;
    updateInputState();

    // 添加用户消息到界面
    addMessage('user', message);

    try {
        let responseStream;

        if (audioBlob) {
            // 音频消息：使用 chat-audio 接口
            responseStream = await sendAudioMessage(audioBlob);
        } else {
            // 文本消息：使用 chat 接口
            responseStream = await sendTextMessage(message);
        }

        // 处理流式响应
        await handleStreamResponse(responseStream);
    } catch (error) {
        console.error('发送消息失败:', error);
        addSystemMessage(`❌ 处理请求时出错: ${error.message}`, 'error');
    } finally {
        state.isProcessing = false;
        updateInputState();
    }
}

async function sendTextMessage(message) {
    const formData = new FormData();
    formData.append('message', message);
    formData.append('user_id', state.currentUser);

    const response = await fetch('/api/chat', {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
    }

    return response.body;
}

async function sendAudioMessage(audioBlob) {
    const formData = new FormData();
    formData.append('audio', audioBlob);
    formData.append('user_id', state.currentUser);

    const response = await fetch('/api/chat-audio', {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
    }

    return response.body;
}

async function handleStreamResponse(stream) {
    const reader = stream.getReader();
    const decoder = new TextDecoder();
    let assistantMessage = null;
    let fullText = '';

    try {
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));

                        switch (data.type) {
                            case 'asr_result':
                                // ASR 识别结果
                                console.log('ASR 结果:', data.text);
                                break;

                            case 'text':
                                // 文本片段
                                if (!assistantMessage) {
                                    assistantMessage = addMessage('assistant', '');
                                }
                                fullText += data.chunk;
                                updateMessageContent(assistantMessage, fullText);
                                break;

                            case 'done':
                                // 响应完成
                                console.log('完整响应:', data.full_text);
                                // 可选：自动播放 TTS
                                if (fullText && assistantMessage) {
                                    await playTTS(fullText);
                                }
                                break;

                            case 'action':
                                // 特殊动作
                                handleAction(data);
                                break;

                            case 'error':
                                // 错误
                                addSystemMessage(`❌ ${data.message}`, 'error');
                                break;
                        }
                    } catch (e) {
                        console.error('解析 SSE 数据失败:', e);
                    }
                }
            }
        }
    } finally {
        reader.releaseLock();
    }
}

function handleAction(data) {
    if (data.action === 'register') {
        if (data.target === 'Unknown_User') {
            addSystemMessage('🎤 请告诉我你的名字？');
        } else {
            addSystemMessage(`🎤 准备注册用户 "${data.target}" 的声纹`);
        }
    }
}

// ==================== 录音功能 ====================
async function startRecording() {
    if (state.isRecording || state.isProcessing) return;

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: 16000,
                channelCount: 1
            }
        });

        state.mediaRecorder = new MediaRecorder(stream, {
            mimeType: 'audio/webm'
        });

        state.audioChunks = [];

        state.mediaRecorder.ondataavailable = (event) => {
            state.audioChunks.push(event.data);
        };

        state.mediaRecorder.onstop = async () => {
            // 转换为 WAV 格式
            const audioBlob = convertToWav(state.audioChunks);
            stream.getTracks().forEach(track => track.stop());

            // 发送消息
            if (audioBlob) {
                await sendMessage('🎤 语音消息', audioBlob);
            }
        };

        state.mediaRecorder.start();
        state.isRecording = true;

        // 显示录音界面
        elements.recordingOverlay.classList.add('active');
        elements.recordBtn.classList.add('recording');
        addSystemMessage('🎙️ 开始录音...');

    } catch (error) {
        console.error('启动录音失败:', error);
        addSystemMessage('❌ 无法访问麦克风，请检查权限设置', 'error');
    }
}

function stopRecording() {
    if (!state.isRecording) return;

    state.mediaRecorder.stop();
    state.isRecording = false;

    // 隐藏录音界面
    elements.recordingOverlay.classList.remove('active');
    elements.recordBtn.classList.remove('recording');
    addSystemMessage('⏹️ 录音完成，正在处理...');
}

function convertToWav(chunks) {
    // 简单转换：将 WebM 转换为 WAV
    // 实际项目中可能需要更复杂的转换逻辑
    const blob = new Blob(chunks, { type: 'audio/webm' });
    return blob;
}

// ==================== 声纹注册 ====================
async function handleRegisterSpeaker() {
    const userId = prompt('请输入用户名:');
    if (!userId) return;

    addSystemMessage(`🎤 准备为 "${userId}" 注册声纹...`);
    addSystemMessage('请说一句话（至少3秒）用于注册声纹...');

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const recorder = new MediaRecorder(stream);
        const chunks = [];

        recorder.ondataavailable = (e) => chunks.push(e.data);

        recorder.onstop = async () => {
            const audioBlob = new Blob(chunks, { type: 'audio/webm' });

            const formData = new FormData();
            formData.append('audio', audioBlob);
            formData.append('user_id', userId);

            try {
                const result = await apiRequest('/api/speaker/register', 'POST', formData, true);
                addSystemMessage(`✅ 声纹注册成功！用户: ${result.user_id}`);
                await loadSpeakers();
            } catch (error) {
                addSystemMessage(`❌ 声纹注册失败: ${error.message}`, 'error');
            }

            stream.getTracks().forEach(track => track.stop());
        };

        recorder.start();

        // 3秒后自动停止录音
        setTimeout(() => {
            recorder.stop();
        }, 3000);

    } catch (error) {
        addSystemMessage(`❌ 无法访问麦克风: ${error.message}`, 'error');
    }
}

// ==================== TTS 播放 ====================
async function playTTS(text) {
    try {
        const formData = new FormData();
        formData.append('text', text);

        const response = await fetch('/api/tts', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('TTS 请求失败');

        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);

        elements.audioPlayer.src = audioUrl;
        elements.audioPlayer.play();

        // 播放完成后清理
        elements.audioPlayer.onended = () => {
            URL.revokeObjectURL(audioUrl);
        };

    } catch (error) {
        console.error('TTS 播放失败:', error);
    }
}

// ==================== UI 更新 ====================
function addMessage(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    // 为用户/助手消息添加头像
    const avatar = role === 'user' ? '👤' : '🤖';
    const header = document.createElement('div');
    header.className = 'message-header';
    header.innerHTML = `<span class="avatar">${avatar}</span>`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = escapeHtml(content);

    messageDiv.appendChild(header);
    messageDiv.appendChild(contentDiv);

    elements.chatMessages.appendChild(messageDiv);
    scrollToBottom();

    return contentDiv;
}

function updateMessageContent(element, content) {
    element.innerHTML = escapeHtml(content);
    scrollToBottom();
}

function addSystemMessage(content, type = 'info') {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message system ${type}`;
    messageDiv.innerHTML = `<div class="message-content">${content}</div>`;
    elements.chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

function scrollToBottom() {
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

function updateStatus(status) {
    elements.statusIndicator.className = 'status-indicator';
    elements.statusDot.className = 'status-dot';

    if (status === 'connected') {
        elements.statusIndicator.classList.add('connected');
        elements.statusText.textContent = '已连接';
    } else {
        elements.statusIndicator.classList.add('disconnected');
        elements.statusText.textContent = '未连接';
    }
}

function updateInputState() {
    const disabled = state.isProcessing;
    elements.messageInput.disabled = disabled;
    elements.sendBtn.disabled = disabled;
    elements.recordBtn.disabled = disabled;
    elements.registerBtn.disabled = disabled;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
