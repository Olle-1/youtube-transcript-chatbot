// Base API URL - change this to your deployed API URL
const API_URL = 'https://your-digitalocean-app-url.ondigitalocean.app';

// DOM elements
const messagesContainer = document.getElementById('messages');
const messageInput = document.getElementById('message-input');
const sendButton = document.getElementById('send-button');
const sourcesContainer = document.getElementById('sources');

// Add event listener to send button
sendButton.addEventListener('click', sendMessage);

// Also send message when pressing Enter
messageInput.addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

// Function to send a message
async function sendMessage() {
    const message = messageInput.value.trim();
    
    if (!message) return; // Don't send empty messages
    
    // Add user message to chat
    addMessage(message, 'user');
    
    // Clear input field
    messageInput.value = '';
    
    // Add loading indicator
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message bot-message';
    loadingDiv.innerHTML = '<p>Thinking...</p>';
    messagesContainer.appendChild(loadingDiv);
    
    try {
        // Send message to API
        const response = await fetch(`${API_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: message
            })
        });
        
        // Remove loading indicator
        messagesContainer.removeChild(loadingDiv);
        
        if (!response.ok) {
            throw new Error('Error connecting to the chatbot');
        }
        
        const data = await response.json();
        
        // Add bot response to chat
        addMessage(data.response, 'bot');
        
        // Display sources if available
        if (data.sources && data.sources.length > 0) {
            displaySources(data.sources);
        }
        
    } catch (error) {
        // Remove loading indicator if still present
        if (messagesContainer.contains(loadingDiv)) {
            messagesContainer.removeChild(loadingDiv);
        }
        
        // Show error message
        addMessage('Sorry, there was an error processing your request. Please try again.', 'bot');
        console.error('Error:', error);
    }
}

// Function to add a message to the chat
function addMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const messagePara = document.createElement('p');
    messagePara.textContent = text;
    messageDiv.appendChild(messagePara);
    
    messagesContainer.appendChild(messageDiv);
    
    // Scroll to bottom of messages
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Function to display sources
function displaySources(sources) {
    // Clear current sources
    sourcesContainer.innerHTML = '';
    
    sources.forEach(source => {
        const sourceDiv = document.createElement('div');
        sourceDiv.className = 'source';
        
        const sourceLink = document.createElement('a');
        sourceLink.href = source.url;
        sourceLink.target = '_blank';
        sourceLink.textContent = source.title;
        
        sourceDiv.appendChild(sourceLink);
        sourcesContainer.appendChild(sourceDiv);
    });
}