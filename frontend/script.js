// frontend/script.js

document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const authContainer = document.getElementById('auth-container');
    const chatContainer = document.getElementById('chat-container');
    const loginForm = document.getElementById('login-form');
    const registerForm = document.getElementById('register-form');
    const loginEmailInput = document.getElementById('login-email');
    const loginPasswordInput = document.getElementById('login-password');
    const loginErrorMsg = document.getElementById('login-error');
    const registerEmailInput = document.getElementById('register-email');
    const registerPasswordInput = document.getElementById('register-password');
    const registerErrorMsg = document.getElementById('register-error');
    const registerSuccessMsg = document.getElementById('register-success');
    const userEmailSpan = document.getElementById('user-email');
    const logoutButton = document.getElementById('logout-button');
    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');
    const messagesDiv = document.getElementById('messages');
    const sessionList = document.getElementById('session-list');
    const newChatButton = document.getElementById('new-chat-button');
    const sourcesList = document.getElementById('sources-list');
    const settingsButton = document.getElementById('settings-button');
    const settingsModal = document.getElementById('settings-modal');
    const closeSettingsButton = document.getElementById('close-settings-button');
    const settingsForm = document.getElementById('settings-form');
    const settingsStatus = document.getElementById('settings-status');


    // --- State ---
    let currentToken = localStorage.getItem('liftingChatToken');
    let currentUser = null;
    let currentSessionId = null;
    let eventSource = null; // For SSE connection

    // --- API Base URL ---
    const API_BASE_URL = ''; // Adjust if backend is elsewhere

    // --- Utility Functions ---
    function showAuthView() {
        authContainer.style.display = 'block';
        chatContainer.style.display = 'none';
        settingsModal.style.display = 'none'; // Ensure modal is hidden
    }

    function showChatView() {
        authContainer.style.display = 'none';
        chatContainer.style.display = 'block';
        settingsModal.style.display = 'none'; // Ensure modal is hidden
        fetchUserInfo();
        fetchChatSessions();
        clearChatArea();
    }

    function clearErrorMessages() {
        loginErrorMsg.textContent = '';
        registerErrorMsg.textContent = '';
        registerSuccessMsg.textContent = '';
        settingsStatus.textContent = ''; // Also clear settings status
    }

    function clearChatArea() {
        messagesDiv.innerHTML = '<div class="message assistant">Select a chat or start a new one.</div>';
        sourcesList.innerHTML = '';
        messageInput.value = '';
        currentSessionId = null;
        document.querySelectorAll('#session-list li.active').forEach(el => el.classList.remove('active'));
        if (eventSource) {
            eventSource.close();
            eventSource = null;
            console.log("Closed existing EventSource connection.");
        }
    }

    // --- Authentication ---
    async function handleLogin(event) {
        event.preventDefault();
        clearErrorMessages();
        const email = loginEmailInput.value;
        const password = loginPasswordInput.value;
        const formData = new URLSearchParams();
        formData.append('username', email);
        formData.append('password', password);

        try {
            const response = await fetch(`${API_BASE_URL}/auth/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                body: formData,
            });
            const data = await response.json();
            if (!response.ok) {
                loginErrorMsg.textContent = data.detail || 'Login failed.';
                throw new Error(data.detail || `HTTP error! status: ${response.status}`);
            }
            currentToken = data.access_token;
            localStorage.setItem('liftingChatToken', currentToken);
            showChatView();
        } catch (error) {
            console.error('Login error:', error);
            if (!loginErrorMsg.textContent) {
                 loginErrorMsg.textContent = 'An error occurred during login.';
            }
        }
    }

    async function handleRegister(event) {
        event.preventDefault();
        clearErrorMessages();
        const email = registerEmailInput.value;
        const password = registerPasswordInput.value;

        try {
            const response = await fetch(`${API_BASE_URL}/auth/register`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password }),
            });
            const data = await response.json();
            if (!response.ok) {
                 registerErrorMsg.textContent = data.detail || 'Registration failed.';
                throw new Error(data.detail || `HTTP error! status: ${response.status}`);
            }
            registerSuccessMsg.textContent = 'Registration successful! Please log in.';
            registerForm.reset();
        } catch (error) {
            console.error('Registration error:', error);
             if (!registerErrorMsg.textContent) {
                 registerErrorMsg.textContent = 'An error occurred during registration.';
             }
        }
    }

    function handleLogout() {
        currentToken = null;
        currentUser = null;
        currentSessionId = null;
        localStorage.removeItem('liftingChatToken');
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }
        showAuthView();
        userEmailSpan.textContent = '';
        sessionList.innerHTML = '';
        clearChatArea();
    }

    async function fetchUserInfo() {
        if (!currentToken) return;
        try {
            const response = await fetch(`${API_BASE_URL}/users/me`, {
                headers: { 'Authorization': `Bearer ${currentToken}` }
            });
            if (!response.ok) {
                if (response.status === 401) handleLogout();
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            currentUser = await response.json();
            userEmailSpan.textContent = currentUser.email;
        } catch (error) {
            console.error('Failed to fetch user info:', error);
        }
    }

    // --- Chat History ---
    async function fetchChatSessions() {
        if (!currentToken) return;
        try {
            const response = await fetch(`${API_BASE_URL}/chat/sessions`, {
                headers: { 'Authorization': `Bearer ${currentToken}` }
            });
            if (!response.ok) {
                 if (response.status === 401) handleLogout();
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const sessions = await response.json();
            displaySessions(sessions);
        } catch (error) {
            console.error('Failed to fetch chat sessions:', error);
            sessionList.innerHTML = '<li>Error loading sessions.</li>';
        }
    }

    function displaySessions(sessions) {
        sessionList.innerHTML = '';
        if (!sessions || sessions.length === 0) {
            sessionList.innerHTML = '<li>No chat history yet.</li>';
            return;
        }
        sessions.forEach(session => {
            const li = document.createElement('li');
            li.textContent = session.title || `Chat ${session.id} (${new Date(session.created_at).toLocaleDateString()})`;
            li.dataset.sessionId = session.id;
            if (session.id === currentSessionId) {
                li.classList.add('active');
            }
            sessionList.appendChild(li);
        });
    }

     async function loadSessionMessages(sessionId) {
        if (!currentToken || !sessionId) return;
        console.log(`Loading messages for session: ${sessionId}`);
        messagesDiv.innerHTML = '<div class="message assistant">Loading messages...</div>';
        sourcesList.innerHTML = '';
        currentSessionId = sessionId;

        document.querySelectorAll('#session-list li').forEach(li => {
            li.classList.toggle('active', li.dataset.sessionId == sessionId);
        });

        if (eventSource) {
            eventSource.close();
            eventSource = null;
            console.log("Closed existing EventSource connection before loading history.");
        }

        try {
            const response = await fetch(`${API_BASE_URL}/chat/sessions/${sessionId}/messages`, {
                headers: { 'Authorization': `Bearer ${currentToken}` }
            });
            if (!response.ok) {
                 if (response.status === 401) handleLogout();
                 if (response.status === 404) {
                     messagesDiv.innerHTML = '<div class="message assistant">Chat session not found.</div>';
                     currentSessionId = null;
                     return;
                 }
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const messages = await response.json();
            displayMessages(messages);
        } catch (error) {
            console.error(`Failed to load messages for session ${sessionId}:`, error);
            messagesDiv.innerHTML = '<div class="message assistant error">Error loading messages.</div>';
        }
    }

    function displayMessages(messages) {
        messagesDiv.innerHTML = '';
        if (!messages || messages.length === 0) {
             messagesDiv.innerHTML = '<div class="message assistant">No messages in this chat yet.</div>';
             return;
        }
        messages.forEach(msg => {
            addMessageToDisplay(msg.role, msg.content, msg.sources);
        });
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }

    function addMessageToDisplay(role, content, sources = null) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', ...role.split(' '));
        content = content.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
        messageElement.innerHTML = `<p>${content.replace(/\n/g, '<br>')}</p>`;
        messagesDiv.appendChild(messageElement);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;

        if (role === 'assistant' && sources && sources.length > 0) {
            displaySources(sources);
        }
    }

    function displaySources(sources) {
        sourcesList.innerHTML = '';
        sources.forEach(source => {
            const li = document.createElement('li');
            const a = document.createElement('a');
            a.href = source.url;
            a.textContent = source.title || source.url;
            a.target = "_blank";
            li.appendChild(a);
            sourcesList.appendChild(li);
        });
    }

    function handleSessionClick(event) {
        if (event.target && event.target.tagName === 'LI') {
            const sessionId = event.target.dataset.sessionId;
            if (sessionId && sessionId != currentSessionId) {
                loadSessionMessages(sessionId);
            }
        }
    }

     function handleNewChat() {
        console.log("Starting new chat...");
        clearChatArea();
        document.querySelectorAll('#session-list li.active').forEach(el => el.classList.remove('active'));
    }

    // --- Chat Interaction (Send Message & SSE) ---
    function handleSendMessage() {
        const query = messageInput.value.trim();
        if (!query || !currentToken) return;

        if (!currentSessionId && messagesDiv.querySelector('.message.assistant')?.textContent.includes('Select a chat')) {
             messagesDiv.innerHTML = '';
        }

        addMessageToDisplay('user', query);
        messageInput.value = '';
        sourcesList.innerHTML = '';

        messageInput.disabled = true;
        sendButton.disabled = true;

        setupSSE(query);
    }

    function setupSSE(query) {
        if (eventSource) {
            eventSource.close();
        }

        const requestBody = { query: query };
        if (currentSessionId) {
            requestBody.session_id = currentSessionId;
        }

        fetch(`${API_BASE_URL}/chat/stream`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${currentToken}`,
                'Accept': 'text/event-stream'
            },
            body: JSON.stringify(requestBody)
        })
        .then(response => {
            if (!response.ok) {
                 if (response.status === 401) handleLogout();
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let assistantMessageElement = null;
            let buffer = '';

            function processText({ done, value }) {
                if (done) {
                    console.log("Stream finished.");
                    messageInput.disabled = false;
                    sendButton.disabled = false;
                    if (!requestBody.session_id) {
                        fetchChatSessions(); // Refresh list if new session was created
                    }
                    return;
                }

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();

                lines.forEach(line => {
                    if (line.startsWith('event: error')) {
                        const errorDataLine = lines.find(l => l.startsWith('data:'));
                        if (errorDataLine) {
                            try {
                                const errorJson = JSON.parse(errorDataLine.substring(5).trim());
                                console.error("SSE Error:", errorJson.content);
                                if (!assistantMessageElement) {
                                     addMessageToDisplay('assistant error', `Error: ${errorJson.content}`);
                                } else {
                                     assistantMessageElement.innerHTML += `<br><span class="error-message">Error: ${errorJson.content}</span>`;
                                }
                            } catch (e) { console.error("Failed to parse SSE error data:", e); }
                        }
                    } else if (line.startsWith('event: done')) {
                         const doneDataLine = lines.find(l => l.startsWith('data:'));
                         if (doneDataLine) {
                             try {
                                 const doneJson = JSON.parse(doneDataLine.substring(5).trim());
                                 if (doneJson.session_id && !currentSessionId) {
                                     currentSessionId = doneJson.session_id;
                                     console.log("New session started with ID:", currentSessionId);
                                     fetchChatSessions(); // Refresh list
                                 }
                             } catch (e) { console.error("Failed to parse SSE done data:", e); }
                         }
                    } else if (line.startsWith('data:')) {
                        const data = line.substring(5).trim();
                        if (data) {
                            try {
                                const chunkData = JSON.parse(data);
                                if (chunkData.content) {
                                    if (!assistantMessageElement) {
                                        const div = document.createElement('div');
                                        div.classList.add('message', 'assistant');
                                        div.innerHTML = `<p></p>`; // Create container first
                                        messagesDiv.appendChild(div);
                                        assistantMessageElement = div.querySelector('p');
                                    }
                                    // Append content, replacing newlines
                                    assistantMessageElement.innerHTML += chunkData.content.replace(/\n/g, '<br>');
                                    messagesDiv.scrollTop = messagesDiv.scrollHeight;
                                }
                                if (chunkData.sources) {
                                     displaySources(chunkData.sources);
                                }
                            } catch (e) {
                                console.error("Failed to parse SSE data chunk:", e, "Data:", data);
                            }
                        }
                    }
                });
                reader.read().then(processText);
            }
            reader.read().then(processText);
        })
        .catch(error => {
            console.error('SSE Fetch Error:', error);
            addMessageToDisplay('assistant error', `Error connecting to chat stream: ${error.message}`);
            messageInput.disabled = false;
            sendButton.disabled = false;
        });
    }

    // --- Profile Settings Modal ---
    function openSettingsModal() {
        settingsModal.style.display = 'flex'; // Use flex to enable centering
        settingsStatus.textContent = ''; // Clear previous status
        loadSettings(); // Load current settings when opening
    }

    function closeSettingsModal() {
        settingsModal.style.display = 'none';
    }

    async function loadSettings() {
        if (!currentToken) return;
        settingsStatus.textContent = 'Loading...';
        settingsStatus.style.color = '#333'; // Reset color
        try {
            const response = await fetch(`${API_BASE_URL}/profile/settings`, {
                headers: { 'Authorization': `Bearer ${currentToken}` }
            });
            if (!response.ok) {
                 if (response.status === 401) handleLogout();
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            // Populate form fields based on data.profile_settings
            // Example for the 'example' field, adapt for actual settings
            const settings = data.profile_settings || {};
            const exampleInput = document.getElementById('setting-example');
            if (exampleInput) {
                exampleInput.value = settings.example || '';
            }
            // Add logic here to populate other settings fields you add to the form
            settingsStatus.textContent = ''; // Clear loading message
        } catch (error) {
            console.error('Failed to load settings:', error);
            settingsStatus.textContent = 'Error loading settings.';
            settingsStatus.style.color = '#dc3545';
        }
    }

    async function saveSettings(event) {
        event.preventDefault();
        if (!currentToken) return;
        settingsStatus.textContent = 'Saving...';
        settingsStatus.style.color = '#333'; // Reset color

        // Gather data from form - adapt for your actual fields
        const settingsData = {};
        // Example: settingsData.example = formData.get('example');
        // Iterate over form elements or use specific IDs
        const exampleInput = document.getElementById('setting-example');
         if (exampleInput) {
             settingsData.example = exampleInput.value;
         }
        // Add logic here to gather other settings fields

        try {
            const response = await fetch(`${API_BASE_URL}/profile/settings`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${currentToken}`
                },
                body: JSON.stringify({ profile_settings: settingsData })
            });

             const data = await response.json(); // Read body even for errors

            if (!response.ok) {
                 if (response.status === 401) handleLogout();
                 settingsStatus.textContent = data.detail || 'Error saving settings.';
                 settingsStatus.style.color = '#dc3545';
                throw new Error(data.detail || `HTTP error! status: ${response.status}`);
            }

            settingsStatus.textContent = 'Settings saved successfully!';
            settingsStatus.style.color = '#28a745';
            // Optionally update currentUser state if needed
            currentUser.profile_settings = data.profile_settings;
            // setTimeout(closeSettingsModal, 1500);

        } catch (error) {
            console.error('Failed to save settings:', error);
             if (!settingsStatus.textContent || settingsStatus.textContent === 'Saving...') {
                 settingsStatus.textContent = 'Error saving settings.';
                 settingsStatus.style.color = '#dc3545';
             }
        }
    }


    // --- Add Event Listeners ---
    loginForm.addEventListener('submit', handleLogin);
    registerForm.addEventListener('submit', handleRegister);
    logoutButton.addEventListener('click', handleLogout); // Added logout listener

    // Chat listeners
    sendButton.addEventListener('click', handleSendMessage);
    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    });
    newChatButton.addEventListener('click', handleNewChat);
    sessionList.addEventListener('click', handleSessionClick);
    settingsButton.addEventListener('click', openSettingsModal);
    closeSettingsButton.addEventListener('click', closeSettingsModal);
    // Also close modal if user clicks outside the content area
    window.addEventListener('click', (event) => {
        if (event.target == settingsModal) {
            closeSettingsModal();
        }
    });
    settingsForm.addEventListener('submit', saveSettings);

}); // End DOMContentLoaded