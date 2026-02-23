(() => {
  const STORAGE_KEY = 'satq_ui_state_v1';
  const API_BASE = (window.location.port === '5500' || window.location.port === '5501')
    ? 'http://localhost:8000'
    : '';

  const elements = {
    submitBtn: document.getElementById('submitBtn'),
    clearCacheBtn: document.getElementById('clearCacheBtn'),
    endpoint: document.getElementById('endpoint'),
    count: document.getElementById('count'),
    countRow: document.getElementById('countRow'),
    userContext: document.getElementById('userContext'),
    status: document.getElementById('status'),
    results: document.getElementById('results'),
  };

  const state = {
    mode: '/create-random-question',
    count: 10,
    userContext: '',
    defaultPaths: {
      featuresFile: '',
      analysisFile: '',
      loaded: false,
    },
    questions: [],
    answers: {},
    runId: null,
    generatedAt: null,
  };

  function apiUrl(path) {
    return `${API_BASE}${path}`;
  }

  function setStatus(text) {
    elements.status.textContent = text;
  }

  function loadState() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw);
      if (!parsed || typeof parsed !== 'object') return;
      state.mode = parsed.mode || state.mode;
      state.count = Number.isFinite(parsed.count) ? parsed.count : state.count;
      state.userContext = parsed.userContext || '';
      state.questions = Array.isArray(parsed.questions) ? parsed.questions : [];
      state.answers = parsed.answers && typeof parsed.answers === 'object' ? parsed.answers : {};
      state.runId = parsed.runId || null;
      state.generatedAt = parsed.generatedAt || null;
    } catch {
      localStorage.removeItem(STORAGE_KEY);
    }
  }

  function saveState() {
    const serializable = {
      mode: state.mode,
      count: state.count,
      userContext: state.userContext,
      questions: state.questions,
      answers: state.answers,
      runId: state.runId,
      generatedAt: state.generatedAt,
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(serializable));
  }

  function syncFormFromState() {
    elements.endpoint.value = state.mode;
    elements.count.value = String(state.count);
    elements.userContext.value = state.userContext;
    syncCountVisibility();
  }

  function syncCountVisibility() {
    const showCount = state.mode === '/create-question-problem-set';
    elements.countRow.style.display = showCount ? 'grid' : 'none';
  }

  async function safeJson(res) {
    const raw = await res.text();
    try {
      return JSON.parse(raw);
    } catch {
      throw new Error('Server returned non-JSON response.');
    }
  }

  async function loadDefaultPaths() {
    setStatus('Loading SAT defaults...');
    elements.submitBtn.disabled = true;

    try {
      const res = await fetch(apiUrl('/testing/default-features-file'));
      const data = await safeJson(res);
      if (!res.ok) {
        throw new Error((data && data.detail) || 'Failed to load default features file');
      }
      const featuresFile = String(data.features_file || '').trim();
      const analysisFile = String(data.analysis_file || '').trim();
      if (!featuresFile) {
        throw new Error('Default features file is missing.');
      }
      state.defaultPaths.featuresFile = featuresFile;
      state.defaultPaths.analysisFile = analysisFile;
      state.defaultPaths.loaded = true;
      elements.submitBtn.disabled = false;
      if (state.questions.length > 0) {
        setStatus(`Restored cached session (${state.questions.length} question${state.questions.length === 1 ? '' : 's'}).`);
      } else {
        setStatus('Ready.');
      }
    } catch (error) {
      state.defaultPaths.featuresFile = 'test_data/sat_questions.features.json';
      state.defaultPaths.analysisFile = 'test_data/sat_questions.analysis.json';
      state.defaultPaths.loaded = true;
      elements.submitBtn.disabled = false;
      if (state.questions.length > 0) {
        setStatus(`Ready (fallback defaults). Restored cached session (${state.questions.length} question${state.questions.length === 1 ? '' : 's'}).`);
      } else {
        setStatus(`Ready (fallback defaults). ${error.message}`);
      }
    }
  }

  function extractQuestions(data) {
    if (!data || !Array.isArray(data.generated_questions)) return [];
    return data.generated_questions;
  }

  function normalizeAnswerText(value) {
    return String(value || '')
      .replace(/^[A-D]\s*[\).:-]\s*/i, '')
      .replace(/\s+/g, ' ')
      .trim()
      .toLowerCase();
  }

  function resolveCorrectIndex(answerRaw, choices) {
    const answer = String(answerRaw || '').trim();
    if (!answer) return null;

    const letterMatch = answer.toUpperCase().match(/^\s*([A-D])(?:\s*[\).:-].*)?$/);
    if (letterMatch) {
      const idx = letterMatch[1].charCodeAt(0) - 65;
      if (idx >= 0 && idx < choices.length) return idx;
    }

    const normalizedAnswer = normalizeAnswerText(answer);
    if (!normalizedAnswer) return null;

    for (let i = 0; i < choices.length; i += 1) {
      if (normalizeAnswerText(choices[i]) === normalizedAnswer) {
        return i;
      }
    }

    return null;
  }

  function normalizeQuestion(question, index) {
    const passage = String(question.prompt || '').trim();
    const stem = String(question.question_text || question.question || '').trim();
    const fallbackTitle = `Question ${index + 1}`;
    const choices = Array.isArray(question.answer_choices)
      ? question.answer_choices
      : Array.isArray(question.choices)
      ? question.choices
      : Array.isArray(question.options)
      ? question.options
      : [];
    const answerRaw = String(question.correct_answer || question.answer || question.correctOption || '').trim();
    const correctIndex = resolveCorrectIndex(answerRaw, choices);
    const answerLetter = correctIndex !== null ? String.fromCharCode(65 + correctIndex) : answerRaw.toUpperCase();
    const explanation = String(question.explanation || 'No explanation provided.');
    return {
      passage,
      stem,
      title: stem || passage || fallbackTitle,
      choices: choices.map((choice) => String(choice)),
      answer: answerLetter,
      answerRaw,
      correctIndex,
      explanation,
    };
  }

  function renderQuestions() {
    elements.results.innerHTML = '';

    if (!state.questions.length) {
      elements.results.innerHTML = '<div class="panel meta">Generate questions to begin.</div>';
      return;
    }

    state.questions.forEach((question, index) => {
      const card = document.createElement('div');
      card.className = 'q';

      const title = document.createElement('div');
      title.innerHTML = `<strong>Q${index + 1}.</strong> ${question.title}`;
      card.appendChild(title);

      if (question.passage && question.stem) {
        const passage = document.createElement('p');
        passage.className = 'meta';
        passage.textContent = question.passage;
        card.appendChild(passage);
      }

      if (question.stem && question.stem !== question.title) {
        const stem = document.createElement('p');
        stem.textContent = question.stem;
        card.appendChild(stem);
      }

      const choicesWrap = document.createElement('div');
      choicesWrap.className = 'choices';

      const storedSelection = state.answers[String(index)] || null;

      question.choices.forEach((choiceText, choiceIndex) => {
        const letter = String.fromCharCode(65 + choiceIndex);
        const isSelected = storedSelection === letter;
        const isCorrect = question.correctIndex === choiceIndex || question.answer === letter;

        const wrap = document.createElement('div');
        wrap.className = 'choice-wrap';

        const btn = document.createElement('button');
        btn.className = 'choice';
        btn.textContent = `${letter}) ${choiceText}`;

        const note = document.createElement('div');
        note.className = 'choice-note';

        const explanation = document.createElement('div');
        explanation.className = 'meta hidden';

        if (storedSelection) {
          btn.disabled = true;
          if (isCorrect) {
            btn.classList.add('correct');
            note.className = 'choice-note ok';
            note.textContent = 'Correct choice.';
          }
          if (isSelected && !isCorrect) {
            btn.classList.add('incorrect');
            note.className = 'choice-note bad';
            note.textContent = 'Not correct.';
            explanation.classList.remove('hidden');
            explanation.textContent = `Explanation: ${question.explanation}`;
          }
          if (isSelected && isCorrect) {
            explanation.classList.remove('hidden');
            explanation.textContent = `Explanation: ${question.explanation}`;
          }
        }

        btn.addEventListener('click', () => {
          if (state.answers[String(index)]) return;
          state.answers[String(index)] = letter;
          saveState();
          renderQuestions();
        });

        wrap.appendChild(btn);
        wrap.appendChild(note);
        wrap.appendChild(explanation);
        choicesWrap.appendChild(wrap);
      });

      card.appendChild(choicesWrap);
      elements.results.appendChild(card);
    });
  }

  async function generateQuestions() {
    if (!state.defaultPaths.loaded || !state.defaultPaths.featuresFile) {
      setStatus('Default SAT data is not ready.');
      return;
    }

    const endpoint = state.mode;
    const payload = {
      features_file: state.defaultPaths.featuresFile,
    };

    if (state.defaultPaths.analysisFile) payload.analysis_file = state.defaultPaths.analysisFile;
    if (state.userContext.trim()) payload.user_context = state.userContext.trim();
    if (endpoint === '/create-question-problem-set') payload.count = state.count;

    elements.submitBtn.disabled = true;
    setStatus('Generating...');

    try {
      const res = await fetch(apiUrl(endpoint), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await safeJson(res);
      if (!res.ok) {
        throw new Error((data && data.detail) || 'Request failed');
      }

      const normalized = extractQuestions(data).map(normalizeQuestion);
      state.questions = normalized;
      state.answers = {};
      state.runId = data.run_id || null;
      state.generatedAt = new Date().toISOString();
      saveState();
      renderQuestions();

      setStatus(`Done. Run: ${state.runId || 'n/a'} | Questions: ${normalized.length}`);
    } catch (error) {
      setStatus(`Error: ${error.message}`);
    } finally {
      elements.submitBtn.disabled = false;
    }
  }

  function clearSession() {
    state.questions = [];
    state.answers = {};
    state.runId = null;
    state.generatedAt = null;
    saveState();
    renderQuestions();
    setStatus('Session cleared.');
  }

  function bindEvents() {
    elements.endpoint.addEventListener('change', () => {
      state.mode = elements.endpoint.value;
      saveState();
      syncCountVisibility();
    });

    elements.count.addEventListener('input', () => {
      const parsed = parseInt(elements.count.value || '10', 10);
      state.count = Number.isFinite(parsed) ? Math.min(100, Math.max(1, parsed)) : 10;
      saveState();
    });

    elements.userContext.addEventListener('input', () => {
      state.userContext = elements.userContext.value;
      saveState();
    });

    elements.submitBtn.addEventListener('click', generateQuestions);
    elements.clearCacheBtn.addEventListener('click', clearSession);
  }

  function init() {
    loadState();
    syncFormFromState();
    bindEvents();
    renderQuestions();
    loadDefaultPaths();
  }

  init();
})();
