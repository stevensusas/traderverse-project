document.getElementById('chat-form').addEventListener('submit', async function(event) {
    event.preventDefault();
    
    const questionInput = document.getElementById('question');
    const question = questionInput.value;
    if (!question) return;

    // Display user message
    displayMessage(question, 'user-message');

    // Display loading indicator
    const loadingMessage = displayMessage('Loading...', 'bot-message');
    
    try {
        const response = await fetch('http://127.0.0.1:5000/ask_question', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question })
        });

        const data = await response.json();
        const botResponse = data.response || 'No response from the bot';

        // Remove loading indicator
        loadingMessage.remove();

        // Display bot response
        displayMessage(botResponse, 'bot-message');
    } catch (error) {
        console.error('Error asking question:', error);
        loadingMessage.textContent = 'Sorry, something went wrong. Please try again later.';
    }

    questionInput.value = '';
});

function displayMessage(message, className) {
    const chatBox = document.getElementById('chat-box');
    const messageElement = document.createElement('div');
    messageElement.className = `chat-message ${className}`;
    messageElement.textContent = message;
    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight;
    return messageElement;
}
