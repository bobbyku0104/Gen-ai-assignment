import time
from typing import Optional, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, Depends
from app.models.schemas import (
    HealthResponse,
    UploadResponse,
    QueryRequest,
    QueryResponse,
    DocumentListResponse,
    DeleteResponse,
    Citation
)
from app.rag.loader import PDFLoader
from app.rag.chunker import DocumentChunker
from app.rag.vector_store import VectorStoreManager
from app.rag.retriever import ContextRetriever
from app.rag.generator import AnswerGenerator
from app.utils.logger import logger

# Initialize Router
router = APIRouter()

# Initialize core RAG singletons to share across routes
try:
    db_manager = VectorStoreManager()
    retriever = ContextRetriever(db_manager)
    generator = AnswerGenerator()
except Exception as e:
    logger.critical(f"Failed to initialize core RAG components: {e}")
    # We raise the exception so FastAPI startup fails early instead of serving broken routes
    raise e

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, Depends
from fastapi.responses import HTMLResponse

# Define premium Frontend dashboard served from root
HTML_DASHBOARD = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cost-Efficient RAG Application</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0b0f19;
            --bg-secondary: rgba(17, 24, 39, 0.7);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --accent-glow: #6366f1;
            --accent-hover: #4f46e5;
            --danger: #ef4444;
            --success: #10b981;
            --card-glass: rgba(31, 41, 55, 0.35);
            --sidebar-width: 320px;
            --content-max-width: 760px;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
            -webkit-font-smoothing: antialiased;
        }

        body {
            background-color: var(--bg-primary);
            color: var(--text-primary);
            height: 100vh;
            overflow: hidden;
            display: flex;
            background-image: 
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.12) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(139, 92, 246, 0.08) 0px, transparent 50%);
        }

        /* Layout Grid */
        .app-container {
            display: grid;
            grid-template-columns: var(--sidebar-width) 1fr;
            width: 100%;
            height: 100%;
        }

        /* Sidebar Styling */
        .sidebar {
            background: rgba(10, 15, 26, 0.85);
            border-right: 1px solid var(--border-color);
            backdrop-filter: blur(16px);
            display: flex;
            flex-direction: column;
            padding: 24px 20px;
            overflow-y: auto;
            gap: 24px;
            height: 100%;
        }

        .logo-section {
            display: flex;
            align-items: center;
            gap: 10px;
            padding-bottom: 8px;
        }

        .logo-icon {
            width: 30px;
            height: 30px;
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            color: white;
            box-shadow: 0 0 16px rgba(99, 102, 241, 0.4);
            font-size: 0.9rem;
        }

        .logo-title {
            font-size: 1.05rem;
            font-weight: 700;
            background: linear-gradient(to right, #ffffff, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* Upload Area */
        .upload-card {
            background: var(--card-glass);
            border: 1.5px dashed rgba(99, 102, 241, 0.3);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .upload-card:hover, .upload-card.dragover {
            border-color: var(--accent-glow);
            background: rgba(99, 102, 241, 0.04);
            box-shadow: 0 0 20px rgba(99, 102, 241, 0.1);
        }

        .upload-icon {
            font-size: 1.8rem;
            margin-bottom: 8px;
            display: block;
        }

        .upload-text {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        .upload-subtext {
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 4px;
        }

        #file-input {
            display: none;
        }

        /* Progress indicator */
        .progress-bar {
            height: 3px;
            width: 0%;
            background: linear-gradient(to right, #6366f1, #8b5cf6);
            position: absolute;
            bottom: 0;
            left: 0;
            transition: width 0.2s ease;
        }

        .upload-status {
            font-size: 0.75rem;
            margin-top: 8px;
            font-weight: 600;
            color: var(--accent-glow);
            display: none;
        }

        /* Document Inventory */
        .doc-section-title {
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
            margin-bottom: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .doc-count-badge {
            background: rgba(255, 255, 255, 0.06);
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.7rem;
            color: var(--text-primary);
        }

        .doc-list {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .doc-item {
            background: var(--card-glass);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 12px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            transition: all 0.2s ease;
        }

        .doc-item:hover {
            border-color: rgba(255, 255, 255, 0.12);
            background: rgba(255, 255, 255, 0.02);
        }

        .doc-info {
            display: flex;
            flex-direction: column;
            gap: 2px;
            max-width: 80%;
        }

        .doc-name {
            font-size: 0.8rem;
            font-weight: 600;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            color: var(--text-primary);
        }

        .doc-meta {
            font-size: 0.7rem;
            color: var(--text-secondary);
            display: flex;
            gap: 6px;
        }

        .doc-delete-btn {
            background: transparent;
            border: none;
            cursor: pointer;
            padding: 4px;
            border-radius: 6px;
            color: var(--text-secondary);
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .doc-delete-btn:hover {
            background: rgba(239, 68, 68, 0.1);
            color: var(--danger);
        }

        .doc-delete-btn svg {
            width: 14px;
            height: 14px;
            fill: currentColor;
        }

        .empty-docs {
            text-align: center;
            padding: 24px 10px;
            font-size: 0.75rem;
            color: var(--text-secondary);
            border: 1px dashed var(--border-color);
            border-radius: 8px;
        }

        /* Workspace Pane */
        .workspace {
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow: hidden;
            position: relative;
        }

        /* Workspace Header */
        .workspace-header {
            height: 60px;
            border-bottom: 1px solid var(--border-color);
            padding: 0 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(11, 15, 25, 0.4);
            backdrop-filter: blur(12px);
            z-index: 10;
        }

        .connection-status {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.75rem;
            font-weight: 500;
            color: var(--text-secondary);
        }

        .status-dot {
            width: 7px;
            height: 7px;
            background: var(--danger);
            border-radius: 50%;
            box-shadow: 0 0 6px var(--danger);
        }

        .status-dot.online {
            background: var(--success);
            box-shadow: 0 0 6px var(--success);
        }

        .perf-banner {
            display: flex;
            gap: 12px;
            font-size: 0.7rem;
            color: var(--text-secondary);
        }

        .perf-item {
            background: rgba(255, 255, 255, 0.03);
            padding: 3px 8px;
            border-radius: 6px;
            border: 1px solid var(--border-color);
        }

        .perf-item span {
            color: var(--accent-glow);
            font-weight: 600;
        }

        /* Chat History Log */
        .chat-history {
            flex: 1;
            overflow-y: auto;
            padding: 32px 16px;
            display: flex;
            flex-direction: column;
            gap: 32px;
            width: 100%;
        }

        /* Custom Scrollbar */
        .chat-history::-webkit-scrollbar, .sidebar::-webkit-scrollbar {
            width: 6px;
        }

        .chat-history::-webkit-scrollbar-thumb, .sidebar::-webkit-scrollbar-thumb {
            background-color: rgba(255, 255, 255, 0.06);
            border-radius: 3px;
        }

        .message-row {
            display: flex;
            width: 100%;
        }

        .message-wrapper {
            max-width: var(--content-max-width);
            width: 100%;
            margin: 0 auto;
            display: flex;
            gap: 16px;
            align-items: flex-start;
        }

        /* User Message Bubble Alignment */
        .message-row.user {
            justify-content: flex-end;
        }

        .message-row.user .message-wrapper {
            justify-content: flex-end;
        }

        .message-row.user .message-bubble {
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid rgba(99, 102, 241, 0.25);
            color: var(--text-primary);
            border-radius: 18px;
            padding: 12px 18px;
            max-width: 70%;
            word-break: break-word;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            font-size: 0.95rem;
            line-height: 1.5;
        }

        /* Assistant Message styling */
        .message-row.assistant .message-bubble {
            width: 100%;
            color: var(--text-primary);
            font-size: 0.95rem;
            line-height: 1.6;
            word-break: break-word;
        }

        /* ChatGPT-style Avatars */
        .avatar {
            width: 28px;
            height: 28px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.75rem;
            flex-shrink: 0;
        }

        .avatar.user-avatar {
            background: #4f46e5;
            color: white;
            order: 2;
            margin-left: 10px;
            display: none;
        }

        .avatar.assistant-avatar {
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            color: white;
            box-shadow: 0 0 10px rgba(99, 102, 241, 0.2);
        }

        /* Sleek Citations Drawer */
        .citations-container {
            margin-top: 14px;
            padding-top: 10px;
            border-top: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .citations-header {
            font-size: 0.75rem;
            font-weight: 700;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .citations-pills {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .citation-pill {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-color);
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.75rem;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .citation-pill:hover {
            background: rgba(99, 102, 241, 0.08);
            border-color: rgba(99, 102, 241, 0.3);
            color: var(--text-primary);
        }

        .citation-drawer {
            display: none;
            background: rgba(0, 0, 0, 0.15);
            border-left: 2px solid var(--accent-glow);
            padding: 10px 12px;
            border-radius: 0 6px 6px 0;
            font-size: 0.8rem;
            color: var(--text-secondary);
            line-height: 1.45;
            margin-top: 4px;
            font-style: italic;
        }

        .citation-drawer.open {
            display: block;
            animation: slideDown 0.2s ease-out;
        }

        @keyframes slideDown {
            from { opacity: 0; transform: translateY(-4px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Typing indicator */
        .typing-bubble {
            display: flex;
            align-items: center;
            gap: 5px;
            padding: 8px 12px !important;
        }

        .typing-dot {
            width: 5px;
            height: 5px;
            background: var(--text-secondary);
            border-radius: 50%;
            animation: bounce 1.4s infinite ease-in-out;
        }

        .typing-dot:nth-child(1) { animation-delay: -0.32s; }
        .typing-dot:nth-child(2) { animation-delay: -0.16s; }

        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }

        /* Centered Input Area */
        .input-area {
            padding: 16px 24px 24px 24px;
            background: linear-gradient(to top, var(--bg-primary) 70%, transparent);
            width: 100%;
        }

        .input-container-centered {
            max-width: var(--content-max-width);
            width: 100%;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .input-wrapper {
            background: rgba(31, 41, 55, 0.45);
            border: 1px solid var(--border-color);
            border-radius: 24px;
            padding: 6px 12px 6px 20px;
            display: flex;
            align-items: center;
            gap: 12px;
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
            backdrop-filter: blur(12px);
            transition: all 0.2s ease;
        }

        .input-wrapper:focus-within {
            border-color: rgba(99, 102, 241, 0.45);
            box-shadow: 0 4px 24px rgba(99, 102, 241, 0.05);
        }

        #query-input {
            flex: 1;
            background: transparent;
            border: none;
            outline: none;
            color: var(--text-primary);
            font-size: 0.95rem;
            padding: 8px 0;
        }

        #query-input::placeholder {
            color: var(--text-secondary);
        }

        .send-btn {
            background: var(--accent-glow);
            border: none;
            width: 34px;
            height: 34px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            cursor: pointer;
            transition: all 0.2s ease;
            flex-shrink: 0;
        }

        .send-btn:hover {
            background: var(--accent-hover);
            transform: scale(1.05);
        }

        .send-btn svg {
            width: 16px;
            height: 16px;
            fill: currentColor;
        }

        .input-footer-text {
            font-size: 0.7rem;
            color: var(--text-secondary);
            text-align: center;
            margin-top: 4px;
        }

        /* Initial Screen */
        .initial-screen {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: var(--text-secondary);
            gap: 16px;
            text-align: center;
        }

        .initial-icon {
            width: 50px;
            height: 50px;
            background: rgba(99, 102, 241, 0.1);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            border: 1px solid rgba(99, 102, 241, 0.2);
            color: var(--accent-glow);
        }

        .initial-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        .initial-desc {
            font-size: 0.85rem;
            max-width: 320px;
            line-height: 1.5;
        }
    </style>
</head>
<body>

    <div class="app-container">
        <!-- Sidebar Panel -->
        <aside class="sidebar">
            <div class="logo-section">
                <div class="logo-icon">R</div>
                <h1 class="logo-title">Cost-Efficient RAG</h1>
            </div>

            <!-- Upload Card -->
            <div class="upload-card" id="drop-zone">
                <div class="progress-bar" id="upload-progress"></div>
                <span class="upload-icon">📁</span>
                <p class="upload-text">Upload PDF Document</p>
                <p class="upload-subtext">Drag & drop or click to browse</p>
                <input type="file" id="file-input" accept=".pdf">
                <div class="upload-status" id="upload-status">Processing chunks...</div>
            </div>

            <!-- Ingested Documents Registry -->
            <div>
                <div class="doc-section-title">
                    <span>Ingested Documents</span>
                    <span class="doc-count-badge" id="doc-count">0</span>
                </div>
                <div class="doc-list" id="document-inventory">
                    <!-- Dynamic rendering -->
                </div>
            </div>
        </aside>

        <!-- Main Workspace Pane -->
        <main class="workspace">
            <header class="workspace-header">
                <div class="connection-status">
                    <div class="status-dot" id="db-status-indicator"></div>
                    <span id="db-status-text">Connecting...</span>
                </div>
                <div class="perf-banner">
                    <div class="perf-item">Retrieval: <span id="perf-retrieval">-</span></div>
                    <div class="perf-item">Total Latency: <span id="perf-latency">-</span></div>
                </div>
            </header>

            <!-- Chat History Panel -->
            <div class="chat-history" id="chat-container">
                <div class="initial-screen" id="welcome-screen">
                    <div class="initial-icon">💡</div>
                    <p class="initial-title">How can I assist you today?</p>
                    <p class="initial-desc">Upload a PDF document in the sidebar, and run questions against its content with grounded citations.</p>
                </div>
            </div>

            <!-- Input area panel -->
            <div class="input-area">
                <div class="input-container-centered">
                    <div class="input-wrapper">
                        <input type="text" id="query-input" placeholder="Message RAG assistant..." autocomplete="off">
                        <button class="send-btn" id="send-button" title="Send Question">
                            <svg viewBox="0 0 24 24">
                                <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
                            </svg>
                        </button>
                    </div>
                    <p class="input-footer-text">Gemini-2.5-flash RAG model. Verify output details against source pages.</p>
                </div>
            </div>
        </main>
    </div>

    <script>
        // DOM Handles
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');
        const uploadProgress = document.getElementById('upload-progress');
        const uploadStatus = document.getElementById('upload-status');
        const documentInventory = document.getElementById('document-inventory');
        const docCount = document.getElementById('doc-count');
        const dbStatusIndicator = document.getElementById('db-status-indicator');
        const dbStatusText = document.getElementById('db-status-text');
        const perfLatency = document.getElementById('perf-latency');
        const perfRetrieval = document.getElementById('perf-retrieval');
        const chatContainer = document.getElementById('chat-container');
        const queryInput = document.getElementById('query-input');
        const sendButton = document.getElementById('send-button');
        const welcomeScreen = document.getElementById('welcome-screen');

        // Startup actions
        document.addEventListener('DOMContentLoaded', () => {
            fetchHealthCheck();
            fetchIngestedDocuments();
            
            // Periodically check health check status every 10 seconds
            setInterval(fetchHealthCheck, 10000);
        });

        // Trigger health status check
        async function fetchHealthCheck() {
            try {
                const res = await fetch('/health');
                const data = await res.json();
                if (data.database_connected) {
                    dbStatusIndicator.className = 'status-dot online';
                    dbStatusText.innerText = 'ChromaDB Connected';
                } else {
                    dbStatusIndicator.className = 'status-dot';
                    dbStatusText.innerText = 'Database Degraded';
                }
            } catch (err) {
                dbStatusIndicator.className = 'status-dot';
                dbStatusText.innerText = 'Connection Lost';
            }
        }

        // Fetch ingested PDF list
        async function fetchIngestedDocuments() {
            try {
                const res = await fetch('/documents');
                const data = await res.json();
                renderDocumentInventory(data.documents);
            } catch (err) {
                console.error("Failed to load documents list", err);
            }
        }

        // Render documents lists in sidebar
        function renderDocumentInventory(docs) {
            documentInventory.innerHTML = '';
            docCount.innerText = docs.length;

            if (docs.length === 0) {
                documentInventory.innerHTML = `
                    <div class="empty-docs">
                        No documents ingested yet.
                    </div>
                `;
                return;
            }

            docs.forEach(doc => {
                const item = document.createElement('div');
                item.className = 'doc-item';
                
                const timeFormatted = new Date(doc.upload_time).toLocaleDateString(undefined, {
                    month: 'short',
                    day: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit'
                });

                item.innerHTML = `
                    <div class="doc-info">
                        <span class="doc-name" title="${doc.filename}">${doc.filename}</span>
                        <div class="doc-meta">
                            <span>Chunks: ${doc.total_chunks}</span>
                            <span>•</span>
                            <span>${timeFormatted}</span>
                        </div>
                    </div>
                    <button class="doc-delete-btn" onclick="deleteDocument('${doc.document_id}')" title="Delete Document">
                        <svg viewBox="0 0 24 24">
                            <path d="M19,4H15.5L14.5,3H9.5L8.5,4H5V6H19M6,19A2,2 0 0,0 8,21H16A2,2 0 0,0 8,21H16A2,2 0 0,0 18,19V7H6V19Z"/>
                        </svg>
                    </button>
                `;
                documentInventory.appendChild(item);
            });
        }

        // Deletion call
        async function deleteDocument(docId) {
            if (!confirm("Are you sure you want to delete this document? This will remove all associated vector chunks.")) {
                return;
            }
            try {
                const res = await fetch(`/documents/${docId}`, { method: 'DELETE' });
                if (res.ok) {
                    fetchIngestedDocuments();
                } else {
                    const err = await res.json();
                    alert("Delete failed: " + err.detail);
                }
            } catch (err) {
                alert("Delete failed.");
            }
        }

        // Drop zone file browsing triggers
        dropZone.addEventListener('click', () => fileInput.click());
        
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                handleFileUpload(fileInput.files[0]);
            }
        });

        // Drag & Drop handlers
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                handleFileUpload(e.dataTransfer.files[0]);
            }
        });

        // Perform Upload form submission
        async function handleFileUpload(file) {
            if (!file.name.endsWith('.pdf')) {
                alert("Error: Only PDF documents are supported.");
                return;
            }

            const formData = new FormData();
            formData.append('file', file);

            uploadProgress.style.width = '0%';
            uploadStatus.innerText = 'Uploading file...';
            uploadStatus.style.display = 'block';

            // Simulate upload progress
            let width = 0;
            const progressInterval = setInterval(() => {
                if (width >= 90) {
                    clearInterval(progressInterval);
                } else {
                    width += 15;
                    uploadProgress.style.width = width + '%';
                }
            }, 100);

            try {
                const res = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                
                clearInterval(progressInterval);
                uploadProgress.style.width = '100%';

                const data = await res.json();
                if (res.ok) {
                    uploadStatus.innerText = 'Processing complete!';
                    setTimeout(() => {
                        uploadStatus.style.display = 'none';
                        uploadProgress.style.width = '0%';
                    }, 2000);
                    fetchIngestedDocuments();
                } else {
                    uploadStatus.innerText = 'Upload failed.';
                    alert("Error: " + data.detail);
                }
            } catch (err) {
                clearInterval(progressInterval);
                uploadProgress.style.width = '0%';
                uploadStatus.innerText = 'Upload failed.';
                alert("Upload failed.");
            }
        }

        // Query submission triggers
        sendButton.addEventListener('click', submitQuery);
        queryInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                submitQuery();
            }
        });

        // Query handler
        async function submitQuery() {
            const question = queryInput.value.trim();
            if (!question) return;

            // Remove welcome screen on first query
            if (welcomeScreen) {
                welcomeScreen.remove();
            }

            // Append user message bubble
            appendMessage(question, 'user');
            queryInput.value = '';

            // Append typing animation bubble
            const typingBubble = appendTypingIndicator();
            chatContainer.scrollTop = chatContainer.scrollHeight;

            const startQueryTime = performance.now();

            try {
                const res = await fetch('/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: question })
                });

                typingBubble.remove();

                if (res.ok) {
                    const data = await res.json();
                    
                    // Render assistant response bubble
                    appendMessage(data.answer, 'assistant', data.citations);
                    
                    // Update Performance statistics
                    perfLatency.innerText = data.latency_ms.toFixed(1) + ' ms';
                    // Retrieval duration estimation
                    const totalDuration = performance.now() - startQueryTime;
                    const retrievalEst = Math.max(10, totalDuration - data.latency_ms);
                    perfRetrieval.innerText = retrievalEst.toFixed(1) + ' ms';
                    
                } else {
                    const err = await res.json();
                    appendMessage("Error: Failed to fetch answer. Details: " + err.detail, 'assistant');
                }
            } catch (err) {
                typingBubble.remove();
                appendMessage("Error: Could not connect to API server.", 'assistant');
            }

            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        // Append query/response text bubbles
        function appendMessage(text, sender, citations = []) {
            const row = document.createElement('div');
            row.className = `message-row ${sender}`;
            
            const wrapper = document.createElement('div');
            wrapper.className = 'message-wrapper';

            // Add avatar icon
            const avatar = document.createElement('div');
            avatar.className = `avatar ${sender}-avatar`;
            avatar.innerText = sender === 'user' ? 'U' : 'AI';
            wrapper.appendChild(avatar);

            const bubble = document.createElement('div');
            bubble.className = 'message-bubble';
            bubble.innerText = text;



            wrapper.appendChild(bubble);
            row.appendChild(wrapper);
            chatContainer.appendChild(row);
        }

        // Typing dynamic dots
        function appendTypingIndicator() {
            const row = document.createElement('div');
            row.className = 'message-row assistant';

            const wrapper = document.createElement('div');
            wrapper.className = 'message-wrapper';

            const avatar = document.createElement('div');
            avatar.className = 'avatar assistant-avatar';
            avatar.innerText = 'AI';
            wrapper.appendChild(avatar);

            const bubble = document.createElement('div');
            bubble.className = 'message-bubble typing-bubble';
            bubble.innerHTML = `
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            `;

            wrapper.appendChild(bubble);
            row.appendChild(wrapper);
            chatContainer.appendChild(row);
            return row;
        }
    </script>
</body>
</html>
"""

@router.get(
    "/",
    summary="Root Dashboard",
    response_class=HTMLResponse
)
async def root():
    """Serves the premium single-page application dashboard directly to web clients."""
    return HTMLResponse(content=HTML_DASHBOARD)


@router.get(
    "/health",
    summary="Health Probe",
    response_model=HealthResponse
)
async def health_check():
    """Checks the database connection and system operational status."""
    db_connected = False
    try:
        # Basic check to see if database client can heartbeat or describe collections
        db_manager.client.heartbeat()
        db_connected = True
    except Exception as e:
        logger.error(f"Health check failed on ChromaDB: {e}")
        
    return HealthResponse(
        status="ok" if db_connected else "degraded",
        environment="production" if not db_connected else "running",
        database_connected=db_connected
    )

@router.post(
    "/upload",
    summary="Upload PDF Document",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED
)
async def upload_document(
    file: UploadFile = File(..., description="PDF file to parse and ingest"),
    chunk_size: Optional[int] = Form(None, description="Optional override for chunk character size"),
    chunk_overlap: Optional[int] = Form(None, description="Optional override for chunk overlap character size")
):
    """
    Parses and ingests a PDF document.
    Implements SHA-256 deduplication and idempotent routing.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only PDF files are supported."
        )

    try:
        # Read file bytes in memory
        file_bytes = await file.read()
        
        # Step 1: Duplicate PDF detection via SHA-256
        file_hash = db_manager.compute_sha256(file_bytes)
        existing_doc = db_manager.get_document_by_hash(file_hash)
        
        if existing_doc:
            logger.info(f"Duplicate document upload blocked. File hash: {file_hash}")
            return UploadResponse(
                message="Document already ingested (Idempotency bypass).",
                document_id=existing_doc["document_id"],
                filename=existing_doc["filename"],
                total_chunks=existing_doc["total_chunks"],
                file_hash=existing_doc["file_hash"]
            )

        # Step 2: Parse PDF using PDFLoader
        pages = PDFLoader.load_pdf_from_bytes(file_bytes, file.filename)
        if not pages:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="PDF contains no extractable text content."
            )

        # Step 3: Split text using DocumentChunker
        chunked_docs = DocumentChunker.chunk_documents(
            pages_data=pages,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # Step 4: Write to Vector Database (ChromaDB)
        document_id = db_manager.add_documents(
            documents=chunked_docs,
            file_hash=file_hash,
            filename=file.filename
        )

        return UploadResponse(
            message="Document uploaded and processed successfully.",
            document_id=document_id,
            filename=file.filename,
            total_chunks=len(chunked_docs),
            file_hash=file_hash
        )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"In-flight upload failure for '{file.filename}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document upload: {str(e)}"
        )

@router.post(
    "/query",
    summary="Query RAG Pipeline",
    response_model=QueryResponse
)
async def query_rag(request: QueryRequest):
    """
    Performs similarity search context retrieval followed by Gemini generation.
    Returns the answer and source page citations with timing latency.
    """
    start_time = time.perf_counter()
    try:
        # Step 1: Retrieve matching chunks
        chunks = retriever.retrieve_context(
            query=request.question,
            top_k=request.top_k,
            metadata_filter=request.metadata_filter
        )

        # Step 2: Generate response with Gemini
        answer = generator.generate_answer(
            question=request.question,
            chunks=chunks
        )

        # Step 3: Format references/citations
        citations = [
            Citation(
                filename=doc.metadata.get("source", "Unknown"),
                page=int(doc.metadata.get("page", 1)),
                text=doc.page_content
            )
            for doc in chunks
        ]

        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"Query API executed successfully in {latency_ms:.2f} ms.")

        return QueryResponse(
            answer=answer,
            citations=citations,
            latency_ms=round(latency_ms, 2)
        )

    except Exception as e:
        logger.error(f"Query execution failure: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during query execution: {str(e)}"
        )

@router.get(
    "/documents",
    summary="List Ingested Documents",
    response_model=DocumentListResponse
)
async def list_documents():
    """Lists all distinct ingested PDF files from vector storage."""
    try:
        documents = db_manager.get_all_documents()
        return DocumentListResponse(documents=documents)
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve document lists: {str(e)}"
        )

@router.delete(
    "/documents/{document_id}",
    summary="Delete Ingested Document",
    response_model=DeleteResponse
)
async def delete_document(document_id: str):
    """Deletes all chunks associated with a specific document hash ID."""
    try:
        success = db_manager.delete_document(document_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID '{document_id}' was not found in the database."
            )
        
        return DeleteResponse(
            message=f"Successfully deleted document ID '{document_id}' and all associated vector chunks.",
            success=True
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Failed to delete document ID '{document_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to complete document deletion: {str(e)}"
        )
