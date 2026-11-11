/**
 * Local Advanced RAG - Embeddable Chatbot Widget
 * Zero-dependency, Shadow-DOM isolated, streaming document-grounded chatbot.
 */
(function () {
  "use strict";

  // Prevent double injection
  if (window.__RAG_CHAT_WIDGET_LOADED__) return;
  window.__RAG_CHAT_WIDGET_LOADED__ = true;

  // Locate the script element to extract data-* attributes
  const currentScript =
    document.currentScript ||
    document.querySelector('script[src*="chat-widget.js"]') ||
    document.querySelector("script[data-api-url]");

  // Default configuration
  const defaultConfig = {
    apiUrl: currentScript?.getAttribute("data-api-url") || "http://127.0.0.1:8000",
    title: currentScript?.getAttribute("data-title") || "Document Assistant",
    subtitle: currentScript?.getAttribute("data-subtitle") || "Privacy-Preserving Local RAG",
    primaryColor: currentScript?.getAttribute("data-primary-color") || "#0284c7",
    position: currentScript?.getAttribute("data-position") || "bottom-right", // 'bottom-right' or 'bottom-left'
    greeting:
      currentScript?.getAttribute("data-greeting") ||
      "Hello! 👋 I'm your local document assistant. Ask me anything about your uploaded documents, and I'll provide verifiable answers with citations.",
    suggestions: currentScript?.getAttribute("data-suggestions")
      ? currentScript.getAttribute("data-suggestions").split("|")
      : [
          "Summarize the key documents",
          "What are the main findings?",
          "Explain the technical specifications",
        ],
    botAvatar: currentScript?.getAttribute("data-bot-avatar") || "🛡️",
    userAvatar: currentScript?.getAttribute("data-user-avatar") || "👤",
    theme: currentScript?.getAttribute("data-theme") || "dark",
    placeholder: currentScript?.getAttribute("data-placeholder") || "Ask a question about your documents...",
  };

  class RAGChatWidget {
    constructor(config = {}) {
      this.config = Object.assign({}, defaultConfig, config);
      this.sessionId = this.getOrCreateSessionId();
      this.isOpen = false;
      this.isGenerating = false;
      this.messages = [];
      this.init();
    }

    getOrCreateSessionId() {
      const storageKey = "rag_widget_session_id";
      try {
        let sid = sessionStorage.getItem(storageKey);
        if (!sid) {
          sid = "widget_session_" + Date.now() + "_" + Math.random().toString(36).substring(2, 8);
          sessionStorage.setItem(storageKey, sid);
        }
        return sid;
      } catch (e) {
        return "widget_session_" + Date.now();
      }
    }

    resetSession() {
      try {
        sessionStorage.removeItem("rag_widget_session_id");
      } catch (e) {}
      this.sessionId = this.getOrCreateSessionId();
      this.messages = [];
      const messagesContainer = this.shadow.querySelector(".rag-messages");
      if (messagesContainer) {
        messagesContainer.innerHTML = "";
        this.addMessage("assistant", this.config.greeting);
        this.renderSuggestions();
      }
    }

    init() {
      // Create root container attached to document body
      this.host = document.createElement("div");
      this.host.id = "rag-chat-widget-root";
      document.body.appendChild(this.host);

      // Attach Shadow DOM for complete CSS and DOM isolation
      this.shadow = this.host.attachShadow({ mode: "open" });

      // Render styles and HTML structure
      this.render();
      this.attachEvents();

      // Show initial greeting
      this.addMessage("assistant", this.config.greeting);
      this.renderSuggestions();

      // Check backend health status
      this.checkHealth();
    }

    render() {
      const posClass = this.config.position === "bottom-left" ? "pos-left" : "pos-right";
      const isDark = this.config.theme !== "light";

      const style = document.createElement("style");
      style.textContent = `
        * {
          box-sizing: border-box;
          margin: 0;
          padding: 0;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        :host {
          --rag-primary: ${this.config.primaryColor};
          --rag-primary-dark: #0369a1;
          --rag-bg-dark: #0f172a;
          --rag-surface: ${isDark ? "#1e293b" : "#ffffff"};
          --rag-surface-subtle: ${isDark ? "#334155" : "#f1f5f9"};
          --rag-border: ${isDark ? "rgba(255, 255, 255, 0.12)" : "rgba(0, 0, 0, 0.1)"};
          --rag-text-main: ${isDark ? "#f8fafc" : "#0f172a"};
          --rag-text-muted: ${isDark ? "#94a3b8" : "#64748b"};
          --rag-bot-bubble: ${isDark ? "#1e293b" : "#f1f5f9"};
          --rag-user-bubble: var(--rag-primary);
          position: fixed;
          z-index: 2147483647;
        }

        /* Floating Launcher Button */
        .rag-launcher {
          position: fixed;
          bottom: 24px;
          ${this.config.position === "bottom-left" ? "left: 24px;" : "right: 24px;"}
          width: 60px;
          height: 60px;
          border-radius: 50%;
          background: linear-gradient(135deg, var(--rag-primary), var(--rag-primary-dark));
          border: none;
          cursor: pointer;
          box-shadow: 0 8px 24px rgba(2, 132, 199, 0.4), 0 2px 6px rgba(0, 0, 0, 0.2);
          display: flex;
          align-items: center;
          justify-content: center;
          transition: transform 0.25s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.25s ease;
          outline: none;
        }

        .rag-launcher:hover {
          transform: scale(1.08);
          box-shadow: 0 12px 28px rgba(2, 132, 199, 0.5), 0 4px 10px rgba(0, 0, 0, 0.25);
        }

        .rag-launcher:active {
          transform: scale(0.95);
        }

        .rag-launcher-icon, .rag-launcher-close {
          width: 28px;
          height: 28px;
          fill: #ffffff;
          transition: transform 0.25s ease, opacity 0.25s ease;
        }

        .rag-launcher-close {
          position: absolute;
          opacity: 0;
          transform: rotate(-90deg) scale(0.7);
        }

        .rag-launcher.is-active .rag-launcher-icon {
          opacity: 0;
          transform: rotate(90deg) scale(0.7);
        }

        .rag-launcher.is-active .rag-launcher-close {
          opacity: 1;
          transform: rotate(0) scale(1);
        }

        .rag-status-dot {
          position: absolute;
          top: 2px;
          right: 2px;
          width: 14px;
          height: 14px;
          border-radius: 50%;
          background: #22c55e;
          border: 2px solid #0f172a;
          box-shadow: 0 0 6px #22c55e;
        }

        .rag-status-dot.offline {
          background: #ef4444;
          box-shadow: 0 0 6px #ef4444;
        }

        /* Chat Window Modal */
        .rag-window {
          position: fixed;
          bottom: 96px;
          ${this.config.position === "bottom-left" ? "left: 24px;" : "right: 24px;"}
          width: 390px;
          max-width: calc(100vw - 32px);
          height: 600px;
          max-height: calc(100vh - 120px);
          background: var(--rag-bg-dark);
          border: 1px solid var(--rag-border);
          border-radius: 20px;
          box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(255, 255, 255, 0.05);
          display: flex;
          flex-direction: column;
          overflow: hidden;
          opacity: 0;
          pointer-events: none;
          transform: translateY(20px) scale(0.96);
          transition: opacity 0.25s cubic-bezier(0.16, 1, 0.3, 1), transform 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .rag-window.is-open {
          opacity: 1;
          pointer-events: auto;
          transform: translateY(0) scale(1);
        }

        /* Header */
        .rag-header {
          padding: 16px 18px;
          background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
          border-bottom: 1px solid var(--rag-border);
          display: flex;
          align-items: center;
          justify-content: space-between;
          backdrop-filter: blur(8px);
        }

        .rag-header-left {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .rag-avatar {
          width: 38px;
          height: 38px;
          border-radius: 12px;
          background: rgba(2, 132, 199, 0.2);
          border: 1px solid rgba(2, 132, 199, 0.4);
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 20px;
          flex-shrink: 0;
        }

        .rag-header-info {
          display: flex;
          flex-direction: column;
        }

        .rag-header-title {
          font-size: 15px;
          font-weight: 700;
          color: var(--rag-text-main);
          letter-spacing: -0.2px;
        }

        .rag-header-subtitle {
          font-size: 11px;
          color: var(--rag-text-muted);
          display: flex;
          align-items: center;
          gap: 6px;
          margin-top: 2px;
        }

        .rag-indicator-pill {
          display: inline-block;
          width: 7px;
          height: 7px;
          border-radius: 50%;
          background: #22c55e;
          box-shadow: 0 0 6px #22c55e;
        }

        .rag-header-actions {
          display: flex;
          align-items: center;
          gap: 6px;
        }

        .rag-btn-icon {
          width: 32px;
          height: 32px;
          border-radius: 8px;
          background: transparent;
          border: none;
          color: var(--rag-text-muted);
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: background 0.15s, color 0.15s;
          outline: none;
        }

        .rag-btn-icon:hover {
          background: rgba(255, 255, 255, 0.1);
          color: var(--rag-text-main);
        }

        .rag-btn-icon svg {
          width: 18px;
          height: 18px;
          fill: currentColor;
        }

        /* Messages Body */
        .rag-messages {
          flex: 1;
          padding: 16px;
          overflow-y: auto;
          display: flex;
          flex-direction: column;
          gap: 14px;
          scroll-behavior: smooth;
        }

        .rag-messages::-webkit-scrollbar {
          width: 6px;
        }

        .rag-messages::-webkit-scrollbar-thumb {
          background: rgba(255, 255, 255, 0.15);
          border-radius: 3px;
        }

        .rag-msg {
          display: flex;
          gap: 10px;
          max-width: 88%;
          animation: rag-fade-in 0.25s ease-out forwards;
        }

        @keyframes rag-fade-in {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }

        .rag-msg.user {
          align-self: flex-end;
          flex-direction: row-reverse;
        }

        .rag-msg.assistant {
          align-self: flex-start;
        }

        .rag-msg-avatar {
          width: 28px;
          height: 28px;
          border-radius: 8px;
          background: rgba(255, 255, 255, 0.08);
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 14px;
          flex-shrink: 0;
          margin-top: 2px;
        }

        .rag-msg-bubble {
          padding: 12px 14px;
          border-radius: 14px;
          font-size: 13.5px;
          line-height: 1.5;
          word-break: break-word;
        }

        .rag-msg.user .rag-msg-bubble {
          background: var(--rag-user-bubble);
          color: #ffffff;
          border-bottom-right-radius: 4px;
        }

        .rag-msg.assistant .rag-msg-bubble {
          background: var(--rag-bot-bubble);
          color: var(--rag-text-main);
          border: 1px solid var(--rag-border);
          border-bottom-left-radius: 4px;
        }

        /* Formatted text inside assistant response */
        .rag-msg-bubble h1,
        .rag-msg-bubble h2,
        .rag-msg-bubble h3,
        .rag-msg-bubble h4 {
          font-size: 14.5px;
          font-weight: 700;
          color: var(--rag-text-main);
          margin: 6px 0 8px 0;
          line-height: 1.35;
        }

        .rag-msg-bubble p {
          margin-bottom: 8px;
        }

        .rag-msg-bubble p:last-child {
          margin-bottom: 0;
        }

        .rag-msg-bubble strong {
          color: #38bdf8;
          font-weight: 600;
        }

        .rag-msg-bubble em {
          font-style: italic;
        }

        .rag-msg-bubble code {
          background: rgba(0, 0, 0, 0.3);
          border: 1px solid rgba(255, 255, 255, 0.1);
          padding: 2px 6px;
          border-radius: 4px;
          font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
          font-size: 12px;
          color: #fca5a5;
        }

        .rag-msg-bubble pre {
          background: #090d16;
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 8px;
          padding: 10px;
          overflow-x: auto;
          margin: 8px 0;
        }

        .rag-msg-bubble pre code {
          background: transparent;
          border: none;
          padding: 0;
          color: #e2e8f0;
          font-size: 12px;
        }

        .rag-msg-bubble ul, .rag-msg-bubble ol {
          margin-left: 18px;
          margin-bottom: 8px;
        }

        .rag-msg-bubble li {
          margin-bottom: 4px;
        }

        /* Citation Badge */
        .rag-citation-tag {
          display: inline-block;
          background: rgba(56, 189, 248, 0.15);
          color: #38bdf8;
          border: 1px solid rgba(56, 189, 248, 0.4);
          border-radius: 4px;
          padding: 1px 6px;
          font-size: 11px;
          font-family: monospace;
          margin: 2px 2px 2px 0;
        }

        /* Sources Accordion */
        .rag-sources-accordion {
          margin-top: 10px;
          border-top: 1px solid rgba(255, 255, 255, 0.1);
          padding-top: 8px;
        }

        .rag-sources-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          font-size: 11.5px;
          font-weight: 600;
          color: #38bdf8;
          cursor: pointer;
          user-select: none;
          padding: 4px 0;
        }

        .rag-sources-header:hover {
          color: #7dd3fc;
        }

        .rag-sources-list {
          display: none;
          flex-direction: column;
          gap: 6px;
          margin-top: 6px;
        }

        .rag-sources-accordion.open .rag-sources-list {
          display: flex;
        }

        .rag-source-item {
          background: rgba(0, 0, 0, 0.25);
          border: 1px solid rgba(56, 189, 248, 0.2);
          border-radius: 6px;
          padding: 8px;
          font-size: 11px;
        }

        .rag-source-tag {
          font-weight: 600;
          color: #38bdf8;
          margin-bottom: 4px;
        }

        .rag-source-snippet {
          color: #94a3b8;
          line-height: 1.4;
          font-style: italic;
        }

        /* Typing Indicator */
        .rag-typing-indicator {
          display: inline-flex;
          align-items: center;
          gap: 4px;
          padding: 4px 6px;
        }

        .rag-typing-dot {
          width: 6px;
          height: 6px;
          border-radius: 50%;
          background: #38bdf8;
          opacity: 0.4;
          animation: rag-pulse 1.4s infinite ease-in-out;
        }

        .rag-typing-dot:nth-child(1) { animation-delay: 0s; }
        .rag-typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .rag-typing-dot:nth-child(3) { animation-delay: 0.4s; }

        @keyframes rag-pulse {
          0%, 80%, 100% { transform: scale(0.8); opacity: 0.3; }
          40% { transform: scale(1.2); opacity: 1; }
        }

        /* Suggestions Pills */
        .rag-suggestions {
          padding: 0 16px 12px;
          display: flex;
          flex-wrap: wrap;
          gap: 6px;
        }

        .rag-suggestion-pill {
          background: rgba(255, 255, 255, 0.06);
          border: 1px solid rgba(255, 255, 255, 0.12);
          border-radius: 12px;
          padding: 6px 12px;
          font-size: 11.5px;
          color: #cbd5e1;
          cursor: pointer;
          transition: background 0.15s, border-color 0.15s, color 0.15s;
          outline: none;
        }

        .rag-suggestion-pill:hover {
          background: rgba(2, 132, 199, 0.2);
          border-color: rgba(2, 132, 199, 0.5);
          color: #ffffff;
        }

        /* Input Container */
        .rag-input-area {
          padding: 12px 16px;
          background: rgba(15, 23, 42, 0.95);
          border-top: 1px solid var(--rag-border);
          display: flex;
          flex-direction: column;
          gap: 6px;
        }

        .rag-input-box {
          display: flex;
          align-items: center;
          gap: 8px;
          background: rgba(30, 41, 59, 0.7);
          border: 1px solid var(--rag-border);
          border-radius: 12px;
          padding: 6px 10px;
          transition: border-color 0.2s, box-shadow 0.2s;
        }

        .rag-input-box:focus-within {
          border-color: var(--rag-primary);
          box-shadow: 0 0 0 2px rgba(2, 132, 199, 0.25);
        }

        .rag-input {
          flex: 1;
          background: transparent;
          border: none;
          outline: none;
          color: var(--rag-text-main);
          font-size: 13.5px;
          line-height: 1.4;
          resize: none;
          max-height: 80px;
          height: 24px;
        }

        .rag-input::placeholder {
          color: var(--rag-text-muted);
        }

        .rag-send-btn {
          width: 32px;
          height: 32px;
          border-radius: 8px;
          background: var(--rag-primary);
          border: none;
          color: #ffffff;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: background 0.15s, transform 0.15s;
          flex-shrink: 0;
          outline: none;
        }

        .rag-send-btn:hover:not(:disabled) {
          background: var(--rag-primary-dark);
          transform: scale(1.05);
        }

        .rag-send-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .rag-send-btn svg {
          width: 16px;
          height: 16px;
          fill: currentColor;
        }

        .rag-footer {
          font-size: 10.5px;
          color: var(--rag-text-muted);
          text-align: center;
          user-select: none;
        }

        /* Mobile full-screen responsive */
        @media (max-width: 480px) {
          .rag-window {
            bottom: 0;
            right: 0;
            left: 0;
            width: 100vw;
            max-width: 100vw;
            height: 100vh;
            max-height: 100vh;
            border-radius: 0;
            border: none;
          }
          .rag-launcher {
            bottom: 16px;
            right: 16px;
          }
        }
      `;

      this.shadow.appendChild(style);

      // Main container
      const wrapper = document.createElement("div");
      wrapper.className = "rag-widget-container";
      wrapper.innerHTML = `
        <!-- Floating Launcher Button -->
        <button class="rag-launcher" aria-label="Open AI Document Chat" title="${this.config.title}">
          <span class="rag-status-dot" title="Local RAG Status"></span>
          <svg class="rag-launcher-icon" viewBox="0 0 24 24">
            <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H6l-2 2V4h16v12z"/>
          </svg>
          <svg class="rag-launcher-close" viewBox="0 0 24 24">
            <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
          </svg>
        </button>

        <!-- Chat Window -->
        <div class="rag-window" role="dialog" aria-modal="true" aria-label="${this.config.title}">
          <!-- Header -->
          <div class="rag-header">
            <div class="rag-header-left">
              <div class="rag-avatar">${this.config.botAvatar}</div>
              <div class="rag-header-info">
                <span class="rag-header-title">${this.config.title}</span>
                <span class="rag-header-subtitle">
                  <span class="rag-indicator-pill"></span>
                  <span class="rag-status-text">Local RAG • Grounded</span>
                </span>
              </div>
            </div>
            <div class="rag-header-actions">
              <button class="rag-btn-icon rag-btn-reset" title="New Conversation / Reset" aria-label="Reset">
                <svg viewBox="0 0 24 24">
                  <path d="M17.65 6.35C16.2 4.9 14.21 4 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08c-.82 2.33-3.04 4-5.65 4-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/>
                </svg>
              </button>
              <button class="rag-btn-icon rag-btn-close" title="Close" aria-label="Close">
                <svg viewBox="0 0 24 24">
                  <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
                </svg>
              </button>
            </div>
          </div>

          <!-- Messages Container -->
          <div class="rag-messages"></div>

          <!-- Suggested Prompts -->
          <div class="rag-suggestions"></div>

          <!-- Input Area -->
          <div class="rag-input-area">
            <div class="rag-input-box">
              <input type="text" class="rag-input" placeholder="${this.config.placeholder}" aria-label="Chat input" />
              <button class="rag-send-btn" title="Send message" aria-label="Send message">
                <svg viewBox="0 0 24 24">
                  <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
                </svg>
              </button>
            </div>
            <div class="rag-footer">
              🔒 Private Offline RAG • Strictly Document-Grounded
            </div>
          </div>
        </div>
      `;

      this.shadow.appendChild(wrapper);
    }

    attachEvents() {
      const launcher = this.shadow.querySelector(".rag-launcher");
      const closeBtn = this.shadow.querySelector(".rag-btn-close");
      const resetBtn = this.shadow.querySelector(".rag-btn-reset");
      const sendBtn = this.shadow.querySelector(".rag-send-btn");
      const input = this.shadow.querySelector(".rag-input");

      launcher.addEventListener("click", () => this.toggle());
      closeBtn.addEventListener("click", () => this.close());
      resetBtn.addEventListener("click", () => {
        if (confirm("Start a new chat session? Previous messages will be cleared.")) {
          this.resetSession();
        }
      });

      sendBtn.addEventListener("click", () => this.handleSend());
      input.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          this.handleSend();
        }
      });
    }

    open() {
      this.isOpen = true;
      const launcher = this.shadow.querySelector(".rag-launcher");
      const windowEl = this.shadow.querySelector(".rag-window");
      launcher.classList.add("is-active");
      windowEl.classList.add("is-open");
      const input = this.shadow.querySelector(".rag-input");
      setTimeout(() => input && input.focus(), 250);
    }

    close() {
      this.isOpen = false;
      const launcher = this.shadow.querySelector(".rag-launcher");
      const windowEl = this.shadow.querySelector(".rag-window");
      launcher.classList.remove("is-active");
      windowEl.classList.remove("is-open");
    }

    toggle() {
      if (this.isOpen) {
        this.close();
      } else {
        this.open();
      }
    }

    async checkHealth() {
      const statusDot = this.shadow.querySelector(".rag-status-dot");
      const statusText = this.shadow.querySelector(".rag-status-text");
      try {
        const resp = await fetch(`${this.config.apiUrl}/health`, { method: "GET" });
        if (resp.ok) {
          const data = await resp.json();
          if (data.ollama_connected) {
            statusDot.classList.remove("offline");
            statusDot.title = `Ollama Connected: ${data.configured_model}`;
            if (statusText) statusText.textContent = `${data.configured_model} • Grounded`;
          } else {
            statusDot.classList.add("offline");
            statusDot.title = "Ollama model disconnected";
          }
        } else {
          statusDot.classList.add("offline");
        }
      } catch (err) {
        statusDot.classList.add("offline");
        statusDot.title = "Backend unreachable at " + this.config.apiUrl;
      }
    }

    renderSuggestions() {
      const suggestionsContainer = this.shadow.querySelector(".rag-suggestions");
      if (!suggestionsContainer) return;
      suggestionsContainer.innerHTML = "";

      if (!this.config.suggestions || this.config.suggestions.length === 0) return;

      this.config.suggestions.forEach((text) => {
        const pill = document.createElement("button");
        pill.className = "rag-suggestion-pill";
        pill.textContent = text.trim();
        pill.addEventListener("click", () => {
          const input = this.shadow.querySelector(".rag-input");
          input.value = text.trim();
          this.handleSend();
        });
        suggestionsContainer.appendChild(pill);
      });
    }

    formatMarkdown(text) {
      if (!text) return "";

      // Protect code blocks before escaping
      const codeBlocks = [];
      let s = text.replace(/```([a-z0-9_-]*)\n([\s\S]*?)```/g, (match, lang, code) => {
        const idx = codeBlocks.length;
        codeBlocks.push(`<pre><code>${code.trim()}</code></pre>`);
        return `__CODE_BLOCK_${idx}__`;
      });

      // Protect inline code
      const inlineCodes = [];
      s = s.replace(/`([^`]+)`/g, (match, code) => {
        const idx = inlineCodes.length;
        inlineCodes.push(`<code>${code}</code>`);
        return `__INLINE_CODE_${idx}__`;
      });

      // Escape raw HTML entities
      s = s
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");

      // Clean robotic phrases
      s = s.replace(/\bthe\s+retriev(?:al|ed)\s+(?:document\s+)?context\s+(?:provides|indicates|shows|outlines)\s+/gi, 'the available documentation outlines ');
      s = s.replace(/\bbased\s+on\s+the\s+retriev(?:al|ed)\s+(?:document\s+)?context[,:]?\s*/gi, 'According to the available documentation, ');
      s = s.replace(/\bthe\s+retriev(?:al|ed)\s+(?:document\s+)?context\b/gi, 'the available documentation');
      s = s.replace(/\bretriev(?:al|ed)\s+(?:document\s+)?context\b/gi, 'available documentation');

      // Strip trailing raw reference dumps
      s = s.replace(/(?:\n+|\s+)(?:citation\s+tags?|references?|sources?)\s*:[\s\S]*$/i, '').trim();

      // Separate Title Case phrase squashed together with intro sentence on line 1
      s = s.replace(/^(#{1,4}\s*)?(?:\*{2})?([A-Z][a-zA-Z0-9\s/&-]{3,70}?)(?:\*{2})?\s*[:\-]?\s+((?:The|Here|According|Based|Below|These|This|Some|Several)\s+[a-z]+.*)$/gm, function (m, h, title, intro) {
        if (title.split(/\s+/).length <= 10) {
          return `### ${title}\n\n${intro}`;
        }
        return m;
      });

      // Detach inline headings
      s = s.replace(/([^#\n])\s*(#{1,4})\s*/g, '$1\n\n$2 ');
      s = s.replace(/^#\s*$/gm, '');

      // Markdown Headings (###, ####, ##, #) -> convert to clean HTML headings without stray hashes
      s = s.replace(/^#{4,}\s*(.+)$/gim, '<h4>$1</h4>');
      s = s.replace(/^#{3}\s*(.+)$/gim, '<h3>$1</h3>');
      s = s.replace(/^#{2}\s*(.+)$/gim, '<h2>$1</h2>');
      s = s.replace(/^#{1}\s*(.+)$/gim, '<h3>$1</h3>');

      // Cleanup any leftover raw hashes anywhere in text: e.g. inline '### Side Heading'
      s = s.replace(/#{2,4}\s*([^#\n<]+)/g, '<strong>$1</strong>');
      s = s.replace(/#{1,4}/g, '');

      // Bold **text**
      s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

      // Italic *text*
      s = s.replace(/\*([^*]+)\*/g, "<em>$1</em>");

      // Citation tags [Source: doc.pdf, Page: 2]
      s = s.replace(
        /\[Source:\s*([^,;\]]+)(?:[,;]\s*(?:Pages?|p\.|pp\.|Slides?|Sheets?|Sections?):?\s*([^\]]+))?\]/gi,
        (match) => `<span class="rag-citation-tag">${match}</span>`
      );

      // Bullet points
      s = s.replace(/^\s*[\-\*]\s+(.+)$/gm, "<li>$1</li>");
      s = s.replace(/(<li>[\s\S]*?<\/li>)/g, "<ul>$1</ul>");

      // Paragraphs
      const paragraphs = s.split(/\n\s*\n/);
      let html = paragraphs
        .map((p) => {
          p = p.trim();
          if (!p) return "";
          if (p.startsWith("<h") || p.startsWith("<pre>") || p.startsWith("<ul>") || p.startsWith("<ol>")) return p;
          return `<p>${p.replace(/\n/g, "<br/>")}</p>`;
        })
        .join("");

      // Restore protected code blocks
      inlineCodes.forEach((code, idx) => {
        html = html.replace(`__INLINE_CODE_${idx}__`, code);
      });
      codeBlocks.forEach((code, idx) => {
        html = html.replace(`__CODE_BLOCK_${idx}__`, code);
      });

      return html;
    }

    addMessage(role, text, sources = null) {
      const messagesContainer = this.shadow.querySelector(".rag-messages");
      const msgEl = document.createElement("div");
      msgEl.className = `rag-msg ${role}`;

      const avatar = document.createElement("div");
      avatar.className = "rag-msg-avatar";
      avatar.textContent = role === "user" ? this.config.userAvatar : this.config.botAvatar;

      const bubble = document.createElement("div");
      bubble.className = "rag-msg-bubble";

      if (role === "assistant") {
        bubble.innerHTML = this.formatMarkdown(text);
        if (sources && sources.length > 0) {
          bubble.appendChild(this.buildSourcesAccordion(sources));
        }
      } else {
        bubble.textContent = text;
      }

      msgEl.appendChild(avatar);
      msgEl.appendChild(bubble);
      messagesContainer.appendChild(msgEl);
      this.scrollToBottom();

      this.messages.push({ role, content: text, sources });
      return bubble;
    }

    buildSourcesAccordion(sources) {
      const accordion = document.createElement("div");
      accordion.className = "rag-sources-accordion";

      const header = document.createElement("div");
      header.className = "rag-sources-header";
      header.innerHTML = `<span>📚 Verified Sources (${sources.length})</span> <span>▾</span>`;

      const list = document.createElement("div");
      list.className = "rag-sources-list";

      sources.forEach((src) => {
        const item = document.createElement("div");
        item.className = "rag-source-item";
        const tag = src.raw_citation || src.citation || `[Source: ${src.filename || "Document"}]`;
        const score = src.similarity_score ? ` (Relevance: ${src.similarity_score.toFixed(2)})` : "";
        item.innerHTML = `
          <div class="rag-source-tag">${tag} ${score}</div>
          <div class="rag-source-snippet">${src.snippet || ""}</div>
        `;
        list.appendChild(item);
      });

      header.addEventListener("click", () => {
        accordion.classList.toggle("open");
        const arrow = header.querySelector("span:last-child");
        arrow.textContent = accordion.classList.contains("open") ? "▴" : "▾";
      });

      accordion.appendChild(header);
      accordion.appendChild(list);
      return accordion;
    }

    showTypingIndicator() {
      const messagesContainer = this.shadow.querySelector(".rag-messages");
      const msgEl = document.createElement("div");
      msgEl.className = "rag-msg assistant rag-loading-msg";

      const avatar = document.createElement("div");
      avatar.className = "rag-msg-avatar";
      avatar.textContent = this.config.botAvatar;

      const bubble = document.createElement("div");
      bubble.className = "rag-msg-bubble";
      bubble.innerHTML = `
        <div class="rag-typing-indicator">
          <span class="rag-typing-dot"></span>
          <span class="rag-typing-dot"></span>
          <span class="rag-typing-dot"></span>
        </div>
      `;

      msgEl.appendChild(avatar);
      msgEl.appendChild(bubble);
      messagesContainer.appendChild(msgEl);
      this.scrollToBottom();
      return msgEl;
    }

    scrollToBottom() {
      const messagesContainer = this.shadow.querySelector(".rag-messages");
      if (messagesContainer) {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
      }
    }

    async handleSend() {
      if (this.isGenerating) return;
      const input = this.shadow.querySelector(".rag-input");
      const sendBtn = this.shadow.querySelector(".rag-send-btn");
      const text = input.value.trim();
      if (!text) return;

      // Hide suggestions on user interaction
      const suggestionsContainer = this.shadow.querySelector(".rag-suggestions");
      if (suggestionsContainer) suggestionsContainer.style.display = "none";

      // Render user message
      this.addMessage("user", text);
      input.value = "";
      this.isGenerating = true;
      sendBtn.disabled = true;

      const loadingMsg = this.showTypingIndicator();

      const payload = {
        question: text,
        session_id: this.sessionId,
      };

      try {
        const streamUrl = `${this.config.apiUrl}/chat/stream`;
        const response = await fetch(streamUrl, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        if (!response.ok) {
          throw new Error(`Server returned HTTP ${response.status}`);
        }

        // Prepare streaming bubble
        loadingMsg.remove();
        const bubble = this.addMessage("assistant", "");
        let accumulatedTokens = "";
        let sources = [];

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n\n");
          buffer = lines.pop(); // keep remainder

          for (const line of lines) {
            const trimmed = line.trim();
            if (trimmed.startsWith("data: ")) {
              try {
                const data = JSON.parse(trimmed.substring(6));
                if (data.type === "sources") {
                  sources = data.data || [];
                } else if (data.type === "token") {
                  accumulatedTokens += data.content || "";
                  bubble.innerHTML = this.formatMarkdown(accumulatedTokens);
                  this.scrollToBottom();
                } else if (data.type === "done") {
                  if (data.citations && data.citations.length > 0) {
                    sources = data.citations;
                  }
                  if (data.answer) {
                    accumulatedTokens = data.answer;
                  }
                } else if (data.type === "error") {
                  accumulatedTokens += `\n\n⚠️ *${data.message || "An error occurred."}*`;
                  bubble.innerHTML = this.formatMarkdown(accumulatedTokens);
                }
              } catch (e) {}
            }
          }
        }

        // Finalize bubble with formatted markdown and sources
        bubble.innerHTML = this.formatMarkdown(accumulatedTokens);
        if (sources && sources.length > 0) {
          bubble.appendChild(this.buildSourcesAccordion(sources));
        }
        this.scrollToBottom();

        // Update message store
        this.messages[this.messages.length - 1].content = accumulatedTokens;
        this.messages[this.messages.length - 1].sources = sources;
      } catch (err) {
        loadingMsg.remove();
        // Attempt fallback to non-streaming /chat/query if streaming failed
        try {
          const fallbackResp = await fetch(`${this.config.apiUrl}/chat/query`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
          if (fallbackResp.ok) {
            const data = await fallbackResp.json();
            const sources = data.source_cards || data.citations || [];
            this.addMessage("assistant", data.answer, sources);
          } else {
            throw err;
          }
        } catch (fallbackErr) {
          this.addMessage(
            "assistant",
            `⚠️ **Connection Error**: Unable to reach backend at \`${this.config.apiUrl}\`.\n\nPlease verify that the local RAG backend server is running and Ollama is online.`
          );
        }
      } finally {
        this.isGenerating = false;
        sendBtn.disabled = false;
        input.focus();
      }
    }
  }

  // Auto-initialize when script loads
  let instance = null;
  function initWidget(userConfig) {
    if (!instance) {
      instance = new RAGChatWidget(userConfig);
    }
    return instance;
  }

  // Expose Global API
  window.RAGChatWidget = {
    init: initWidget,
    open: () => instance && instance.open(),
    close: () => instance && instance.close(),
    toggle: () => instance && instance.toggle(),
    reset: () => instance && instance.resetSession(),
    send: (query) => {
      if (instance) {
        instance.open();
        const input = instance.shadow.querySelector(".rag-input");
        if (input) {
          input.value = query;
          instance.handleSend();
        }
      }
    },
    getInstance: () => instance,
  };

  // Run automatically if DOM is ready, otherwise wait for DOMContentLoaded
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => initWidget());
  } else {
    initWidget();
  }
})();

