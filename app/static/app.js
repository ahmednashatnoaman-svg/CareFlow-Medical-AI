/**
 * CareFlow Medical AI — Dual-Mode Frontend Client
 * Mode 1: Graph RAG Triage & SOCRATES Engine
 * Mode 2: WHO Guidelines Dialogue Vector RAG
 */

class MedicalApp {
  constructor() {
    this.currentMode = 'triage'; // 'triage' | 'dialogue'
    this.currentLang = 'en'; // 'en' | 'ar'
    this.sessionId = this.generateUUID();
    this.dialogueHistory = [];
    this.isTriageComplete = false;
    this.isLoading = false;

    this.initElements();
    this.bindEvents();
    this.startSession();
  }

  generateUUID() {
    return 'sess_' + Math.random().toString(36).substring(2, 11) + '_' + Date.now();
  }

  initElements() {
    // Mode tabs
    this.tabTriage = document.getElementById('tab-triage');
    this.tabDialogue = document.getElementById('tab-dialogue');
    this.triageSidebar = document.getElementById('triage-sidebar');
    this.dialogueSidebar = document.getElementById('dialogue-sidebar');

    // Language buttons
    this.langEnBtn = document.getElementById('lang-en-btn');
    this.langArBtn = document.getElementById('lang-ar-btn');
    this.resetBtn = document.getElementById('reset-session-btn');

    // Chat UI
    this.messagesContainer = document.getElementById('messages-container');
    this.chatForm = document.getElementById('chat-form');
    this.userInput = document.getElementById('user-input');
    this.triageOptionsBar = document.getElementById('triage-options-bar');
    this.optionsButtons = document.getElementById('options-buttons');

    // Triage State Indicators
    this.turnBadge = document.getElementById('turn-badge');
    this.socratesScoreText = document.getElementById('socrates-score-text');
    this.socratesProgressFill = document.getElementById('socrates-progress-fill');
    this.socratesSlots = document.getElementById('socrates-slots');
    this.confirmedSymptomsList = document.getElementById('confirmed-symptoms-list');
    this.deniedSymptomsList = document.getElementById('denied-symptoms-list');
    this.posCount = document.getElementById('pos-count');
    this.negCount = document.getElementById('neg-count');
    this.statEntropy = document.getElementById('stat-entropy');
    this.statMargin = document.getElementById('stat-margin');
  }

  bindEvents() {
    // Mode Switch
    this.tabTriage.addEventListener('click', () => this.switchMode('triage'));
    this.tabDialogue.addEventListener('click', () => this.switchMode('dialogue'));

    // Language Switch
    this.langEnBtn.addEventListener('click', () => this.switchLanguage('en'));
    this.langArBtn.addEventListener('click', () => this.switchLanguage('ar'));

    // Reset Chat
    this.resetBtn.addEventListener('click', () => this.resetSession());

    // Submit Form
    this.chatForm.addEventListener('submit', (e) => {
      e.preventDefault();
      this.handleUserSubmit();
    });

    // Enter to Send
    this.userInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.handleUserSubmit();
      }
    });

    // Suggested query chips
    document.querySelectorAll('.query-chip').forEach((chip) => {
      chip.addEventListener('click', (e) => {
        const prompt = chip.getAttribute('data-prompt');
        if (prompt) {
          this.userInput.value = prompt;
          this.handleUserSubmit();
        }
      });
    });
  }

  switchMode(mode) {
    if (this.currentMode === mode) return;
    this.currentMode = mode;

    if (mode === 'triage') {
      this.tabTriage.classList.add('active');
      this.tabDialogue.classList.remove('active');
      this.triageSidebar.classList.remove('hidden');
      this.dialogueSidebar.classList.add('hidden');
      this.userInput.placeholder = this.currentLang === 'ar'
        ? 'اكتب الأعراض التي تشعر بها أو اختر رقماً...'
        : 'Describe your symptoms or type an option number...';
    } else {
      this.tabDialogue.classList.add('active');
      this.tabTriage.classList.remove('active');
      this.dialogueSidebar.classList.remove('hidden');
      this.triageSidebar.classList.add('hidden');
      this.triageOptionsBar.classList.add('hidden');
      this.userInput.placeholder = this.currentLang === 'ar'
        ? 'اسأل سؤالاً طبياً حول إرشادات منظمة الصحة العالمية...'
        : 'Ask a medical question grounded in WHO guidelines...';
    }

    this.renderSystemNotice(
      mode === 'triage'
        ? '🩺 Switched to Mode 1: Graph RAG Diagnostic Triage'
        : '📚 Switched to Mode 2: WHO Guidelines Dialogue Assistant'
    );
  }

  switchLanguage(lang) {
    if (this.currentLang === lang) return;
    this.currentLang = lang;

    if (lang === 'ar') {
      this.langArBtn.classList.add('active');
      this.langEnBtn.classList.remove('active');
      document.documentElement.setAttribute('dir', 'rtl');
      document.documentElement.setAttribute('lang', 'ar');
    } else {
      this.langEnBtn.classList.add('active');
      this.langArBtn.classList.remove('active');
      document.documentElement.setAttribute('dir', 'ltr');
      document.documentElement.setAttribute('lang', 'en');
    }

    this.resetSession();
  }

  async startSession() {
    this.messagesContainer.innerHTML = '';
    this.sessionId = this.generateUUID();
    this.isTriageComplete = false;
    this.dialogueHistory = [];

    if (this.currentMode === 'triage') {
      try {
        this.showTypingIndicator();
        const res = await fetch('/api/v1/triage/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: this.sessionId, language: this.currentLang }),
        });
        const data = await res.json();
        this.hideTypingIndicator();

        this.appendAssistantMessage(data.message);
        this.updateTriageDashboard(data);
      } catch (err) {
        this.hideTypingIndicator();
        this.appendAssistantMessage(
          this.currentLang === 'ar'
            ? 'أهلاً بك في خدمة الفحص الذكي. ما هي الأعراض التي تشتكي منها؟'
            : 'Hello! I am your clinical triage assistant. What symptoms are you experiencing today?'
        );
      }
    } else {
      const welcome = this.currentLang === 'ar'
        ? 'مرحباً بك. أنا مساعدك الطبي المبني على إرشادات منظمة الصحة العالمية (WHO Guidelines). كيف يمكنني مساعدتك اليوم؟'
        : 'Welcome. I am your medical assistant grounded in official WHO Clinical Guidelines. How can I help you today?';
      this.appendAssistantMessage(welcome);
    }
  }

  async resetSession() {
    await this.startSession();
  }

  async handleUserSubmit() {
    const text = this.userInput.value.trim();
    if (!text || this.isLoading) return;

    this.userInput.value = '';
    this.appendUserMessage(text);
    this.triageOptionsBar.classList.add('hidden');

    if (this.currentMode === 'triage') {
      await this.sendTriageStep(text);
    } else {
      await this.sendDialogueChat(text);
    }
  }

  async sendTriageOption(optionText, optionIdx) {
    if (this.isLoading) return;
    this.appendUserMessage(optionText);
    this.triageOptionsBar.classList.add('hidden');
    await this.sendTriageStep(String(optionIdx));
  }

  async sendTriageStep(message) {
    this.isLoading = true;
    this.showTypingIndicator();

    try {
      const res = await fetch('/api/v1/triage/step', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: this.sessionId,
          message: message,
          language: this.currentLang,
        }),
      });

      const data = await res.json();
      this.hideTypingIndicator();
      this.isLoading = false;

      this.appendAssistantMessage(data.message);
      this.updateTriageDashboard(data);

      if (data.is_complete && data.diagnostic_report) {
        this.isTriageComplete = true;
        this.triageOptionsBar.classList.add('hidden');
        this.renderDiagnosticReport(data.diagnostic_report);
      } else if (data.options && data.options.length > 0) {
        this.renderTriageOptions(data.options);
      }
    } catch (err) {
      this.hideTypingIndicator();
      this.isLoading = false;
      this.appendAssistantMessage('Sorry, a connection error occurred. Please try again.');
    }
  }

  async sendDialogueChat(query) {
    this.isLoading = true;
    this.showTypingIndicator();

    try {
      const res = await fetch('/api/v1/dialogue/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query,
          top_k: 5,
          conversation_history: this.dialogueHistory,
        }),
      });

      const data = await res.json();
      this.hideTypingIndicator();
      this.isLoading = false;

      this.dialogueHistory.push({ role: 'user', content: query });
      this.dialogueHistory.push({ role: 'assistant', content: data.answer });

      this.appendAssistantDialogueMessage(data.answer, data.sources);
    } catch (err) {
      this.hideTypingIndicator();
      this.isLoading = false;
      this.appendAssistantMessage('Failed to retrieve guidelines. Please check your network or Qdrant credentials.');
    }
  }

  updateTriageDashboard(data) {
    // Turn badge
    if (this.turnBadge) {
      this.turnBadge.textContent = `Turn ${data.turn_count || 0}/${data.max_turns || 8}`;
    }

    // SOCRATES Score & Progress
    const score = data.socrates_score || 0;
    if (this.socratesScoreText) {
      this.socratesScoreText.textContent = `${score} / 8`;
    }
    if (this.socratesProgressFill) {
      const pct = Math.min((score / 8) * 100, 100);
      this.socratesProgressFill.style.width = `${pct}%`;
    }

    // SOCRATES Grid Chips
    if (data.socrates_tracker) {
      Object.keys(data.socrates_tracker).forEach((slot) => {
        const chip = document.querySelector(`.socrates-chip[data-slot="${slot}"]`);
        if (chip) {
          if (data.socrates_tracker[slot]) {
            chip.classList.add('active');
          } else {
            chip.classList.remove('active');
          }
        }
      });
    }

    // Confirmed Symptoms
    if (this.confirmedSymptomsList) {
      const pos = data.positive_symptoms || [];
      this.posCount.textContent = pos.length;
      if (pos.length === 0) {
        this.confirmedSymptomsList.innerHTML = '<span class="empty-state">No symptoms reported yet</span>';
      } else {
        this.confirmedSymptomsList.innerHTML = pos
          .map((s) => `<span class="tag-pill pos">✓ ${this.escapeHtml(s)}</span>`)
          .join('');
      }
    }

    // Denied Symptoms
    if (this.deniedSymptomsList) {
      const neg = data.negated_symptoms || [];
      this.negCount.textContent = neg.length;
      if (neg.length === 0) {
        this.deniedSymptomsList.innerHTML = '<span class="empty-state">None ruled out yet</span>';
      } else {
        this.deniedSymptomsList.innerHTML = neg
          .map((s) => `<span class="tag-pill neg">✗ ${this.escapeHtml(s)}</span>`)
          .join('');
      }
    }

    // Stats
    if (data.stats) {
      if (this.statEntropy) this.statEntropy.textContent = (data.stats.entropy ?? 1.0).toFixed(2);
      if (this.statMargin) this.statMargin.textContent = (data.stats.margin ?? 0.0).toFixed(2);
    }
  }

  renderTriageOptions(options) {
    this.optionsButtons.innerHTML = '';
    options.forEach((opt, idx) => {
      const btn = document.createElement('button');
      btn.className = 'option-btn';
      btn.innerHTML = `<span class="option-num">${idx + 1}</span> <span>${this.escapeHtml(opt)}</span>`;
      btn.addEventListener('click', () => this.sendTriageOption(opt, idx + 1));
      this.optionsButtons.appendChild(btn);
    });
    this.triageOptionsBar.classList.remove('hidden');
    this.scrollToBottom();
  }

  renderDiagnosticReport(report) {
    const card = document.createElement('div');
    card.className = 'diagnostic-report-card';

    let ddxHtml = '';
    (report.top_diagnoses || []).forEach((d) => {
      const urgencyClass = (d.urgency_level || 'urgent').toLowerCase();
      ddxHtml += `
        <div class="ddx-item">
          <div class="ddx-top-row">
            <span class="ddx-name">${this.escapeHtml(d.diagnosis)}</span>
            <span class="ddx-prob">${this.escapeHtml(d.estimated_probability)}</span>
          </div>
          <p class="ddx-reasoning">${this.escapeHtml(d.reasoning)}</p>
          <div class="ddx-evidence-tag">🔗 ${this.escapeHtml(d.graph_evidence)}</div>
        </div>
      `;
    });

    const topUrgency = report.top_diagnoses && report.top_diagnoses[0] ? report.top_diagnoses[0].urgency_level : 'Urgent';
    const urgencyBadgeClass = topUrgency.toLowerCase().includes('emergency') ? 'emergency' : (topUrgency.toLowerCase().includes('routine') ? 'routine' : 'urgent');

    card.innerHTML = `
      <div class="report-header">
        <h4>📋 Differential Diagnosis Summary (For Doctor)</h4>
        <span class="urgency-badge ${urgencyBadgeClass}">${this.escapeHtml(topUrgency)}</span>
      </div>
      <div class="ddx-list">
        ${ddxHtml}
      </div>
      <div class="recommendation-box">
        <strong>Triage Recommendation:</strong> ${this.escapeHtml(report.triage_recommendation || 'Consult physician for definitive care.')}
      </div>
    `;

    this.messagesContainer.appendChild(card);
    this.scrollToBottom();
  }

  appendUserMessage(text) {
    const row = document.createElement('div');
    row.className = 'message-row user';
    row.innerHTML = `
      <div class="msg-avatar">👤</div>
      <div class="msg-bubble">${this.escapeHtml(text)}</div>
    `;
    this.messagesContainer.appendChild(row);
    this.scrollToBottom();
  }

  appendAssistantMessage(text) {
    const row = document.createElement('div');
    row.className = 'message-row assistant';
    const formatted = typeof marked !== 'undefined' ? marked.parse(text) : this.escapeHtml(text);
    row.innerHTML = `
      <div class="msg-avatar">🩺</div>
      <div class="msg-bubble">${formatted}</div>
    `;
    this.messagesContainer.appendChild(row);
    this.scrollToBottom();
  }

  appendAssistantDialogueMessage(answer, sources) {
    const row = document.createElement('div');
    row.className = 'message-row assistant';

    const formattedAnswer = typeof marked !== 'undefined' ? marked.parse(answer) : this.escapeHtml(answer);

    let sourcesHtml = '';
    if (sources && sources.length > 0) {
      const cardsHtml = sources
        .map(
          (s) => `
        <div class="source-card">
          <div class="source-title-row">
            <span>📄 ${this.escapeHtml(s.source_file)}</span>
            <span class="source-score">Match ${(s.relevance_score * 100).toFixed(1)}%</span>
          </div>
          <div style="font-size:11px; color:#06b6d4; margin-bottom:4px;">Section: ${this.escapeHtml(s.section)}</div>
          <p class="source-snippet">${this.escapeHtml(s.snippet)}</p>
        </div>
      `
        )
        .join('');

      sourcesHtml = `
        <div class="sources-accordion">
          <button class="sources-toggle" onclick="this.nextElementSibling.classList.toggle('hidden')">
            📚 View ${sources.length} Grounded WHO Guideline Sources ▾
          </button>
          <div class="sources-list hidden">
            ${cardsHtml}
          </div>
        </div>
      `;
    }

    row.innerHTML = `
      <div class="msg-avatar">📚</div>
      <div class="msg-bubble">
        ${formattedAnswer}
        ${sourcesHtml}
      </div>
    `;

    this.messagesContainer.appendChild(row);
    this.scrollToBottom();
  }

  renderSystemNotice(text) {
    const notice = document.createElement('div');
    notice.style.textAlign = 'center';
    notice.style.fontSize = '12px';
    notice.style.color = '#94a3b8';
    notice.style.margin = '10px 0';
    notice.innerHTML = `<em>${this.escapeHtml(text)}</em>`;
    this.messagesContainer.appendChild(notice);
    this.scrollToBottom();
  }

  showTypingIndicator() {
    this.hideTypingIndicator();
    const row = document.createElement('div');
    row.id = 'typing-indicator-row';
    row.className = 'message-row assistant';
    row.innerHTML = `
      <div class="msg-avatar">⏳</div>
      <div class="msg-bubble typing-bubble">
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
      </div>
    `;
    this.messagesContainer.appendChild(row);
    this.scrollToBottom();
  }

  hideTypingIndicator() {
    const el = document.getElementById('typing-indicator-row');
    if (el) el.remove();
  }

  scrollToBottom() {
    setTimeout(() => {
      this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }, 50);
  }

  escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }
}

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
  window.app = new MedicalApp();
});
