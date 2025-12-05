/**
 * Example: How to use AI-LLM API from JavaScript/Node.js
 */

const BASE_URL = 'http://localhost:8000';

// Using fetch API
async function healthCheck() {
    const response = await fetch(`${BASE_URL}/health`);
    const data = await response.json();
    console.log('Health:', data);
    return data;
}

async function transcribeFile(audioPath) {
    const response = await fetch(`${BASE_URL}/transcribe`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ audio_path: audioPath }),
    });
    const data = await response.json();
    return data;
}

async function transcribeUpload(filePath) {
    const FormData = require('form-data');
    const fs = require('fs');
    
    const form = new FormData();
    form.append('file', fs.createReadStream(filePath));
    
    const response = await fetch(`${BASE_URL}/transcribe/upload`, {
        method: 'POST',
        body: form,
    });
    const data = await response.json();
    return data;
}

async function askQuestion(query, topK = 5) {
    const response = await fetch(`${BASE_URL}/ask`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: query, top_k: topK }),
    });
    const data = await response.json();
    return data;
}

// Example usage
async function main() {
    try {
        // Health check
        const health = await healthCheck();
        console.log('API Status:', health.status);
        
        // Transcribe
        const transcript = await transcribeFile('data/raw/audio/why-hello-there-103596.wav');
        console.log('Transcribed:', transcript.text);
        
        // Ask question
        const answer = await askQuestion('What is the main topic?');
        console.log('Answer:', answer.answer);
        console.log('Citations:', answer.contexts.length);
    } catch (error) {
        console.error('Error:', error);
    }
}

// Browser version (no file system)
async function transcribeUploadBrowser(fileInput) {
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    const response = await fetch(`${BASE_URL}/transcribe/upload`, {
        method: 'POST',
        body: formData,
    });
    const data = await response.json();
    return data;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        healthCheck,
        transcribeFile,
        transcribeUpload,
        askQuestion,
        transcribeUploadBrowser,
    };
}

