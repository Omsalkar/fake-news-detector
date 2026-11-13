"""
Fake News Detection - One Page UI
Run with: python app.py
"""

from flask import Flask, request, jsonify, render_template_string
import pickle
import os
import re
import json

app = Flask(__name__)

# ─── HTML TEMPLATE ────────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>FakeWatch — News Verifier</title>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:ital,wght@0,300;0,500;1,300&display=swap" rel="stylesheet"/>
<style>
  :root {
    --ink: #0d0d0d;
    --paper: #f5f0e8;
    --red: #c0392b;
    --green: #1a6b3c;
    --yellow: #e8c84a;
    --muted: #8a8070;
    --border: #2a2a2a;
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--paper);
    color: var(--ink);
    font-family: 'IBM Plex Sans', sans-serif;
    min-height: 100vh;
    background-image:
      repeating-linear-gradient(0deg, transparent, transparent 27px, rgba(0,0,0,0.04) 27px, rgba(0,0,0,0.04) 28px);
  }

  /* HEADER */
  header {
    border-bottom: 4px solid var(--ink);
    padding: 18px 40px 14px;
    display: flex;
    align-items: baseline;
    gap: 18px;
    background: var(--paper);
    position: sticky;
    top: 0;
    z-index: 100;
  }
  header h1 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.2rem;
    letter-spacing: 0.04em;
    line-height: 1;
  }
  header .edition {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    border: 1px solid var(--muted);
    padding: 2px 8px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
  }
  header .status-dot {
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
  }
  .dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--green);
    animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }

  /* MAIN GRID */
  main {
    max-width: 1100px;
    margin: 0 auto;
    padding: 40px 24px;
    display: grid;
    grid-template-columns: 1fr 340px;
    gap: 32px;
  }

  /* SECTION LABELS */
  .section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid #ccc;
    padding-bottom: 6px;
    margin-bottom: 16px;
  }

  /* INPUT PANEL */
  .input-panel {}

  textarea {
    width: 100%;
    min-height: 220px;
    background: transparent;
    border: 2px solid var(--ink);
    padding: 16px;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 1rem;
    line-height: 1.6;
    resize: vertical;
    outline: none;
    color: var(--ink);
    transition: border-color 0.2s;
  }
  textarea:focus { border-color: var(--red); }
  textarea::placeholder { color: var(--muted); font-style: italic; }

  .char-count {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    text-align: right;
    margin-top: 6px;
  }

  /* SAMPLE BUTTONS */
  .samples {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin: 14px 0;
  }
  .sample-btn {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    padding: 5px 12px;
    border: 1px solid var(--ink);
    background: transparent;
    cursor: pointer;
    transition: all 0.15s;
    letter-spacing: 0.05em;
  }
  .sample-btn:hover { background: var(--ink); color: var(--paper); }

  /* ANALYSE BUTTON */
  .analyse-btn {
    display: block;
    width: 100%;
    margin-top: 16px;
    padding: 16px;
    background: var(--ink);
    color: var(--paper);
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.6rem;
    letter-spacing: 0.12em;
    border: none;
    cursor: pointer;
    transition: background 0.2s, transform 0.1s;
    position: relative;
    overflow: hidden;
  }
  .analyse-btn:hover { background: var(--red); }
  .analyse-btn:active { transform: scale(0.99); }
  .analyse-btn.loading { pointer-events: none; }
  .analyse-btn .spinner {
    display: none;
    width: 18px; height: 18px;
    border: 2px solid rgba(255,255,255,0.3);
    border-top-color: #fff;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    position: absolute; right: 20px; top: 50%; transform: translateY(-50%);
  }
  .analyse-btn.loading .spinner { display: block; }
  @keyframes spin { to { transform: translateY(-50%) rotate(360deg); } }

  /* VERDICT PANEL */
  .verdict-panel {}
  .verdict-box {
    border: 3px solid var(--ink);
    padding: 24px;
    text-align: center;
    position: relative;
    min-height: 160px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 10px;
    transition: border-color 0.3s, background 0.3s;
  }
  .verdict-box.real   { border-color: var(--green); background: rgba(26,107,60,0.05); }
  .verdict-box.fake   { border-color: var(--red);   background: rgba(192,57,43,0.05); }
  .verdict-box.waiting { border-color: #ccc; }

  .verdict-stamp {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.4rem;
    letter-spacing: 0.08em;
    line-height: 1;
    opacity: 0;
    transform: scale(1.4) rotate(-4deg);
    transition: opacity 0.4s, transform 0.4s;
  }
  .verdict-box.real .verdict-stamp   { color: var(--green); }
  .verdict-box.fake .verdict-stamp   { color: var(--red);   }
  .verdict-box.shown .verdict-stamp  { opacity: 1; transform: scale(1) rotate(-4deg); }

  .verdict-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    letter-spacing: 0.1em;
  }

  /* CONFIDENCE BAR */
  .conf-wrap { width: 100%; margin-top: 8px; }
  .conf-label {
    display: flex;
    justify-content: space-between;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    margin-bottom: 4px;
  }
  .conf-track {
    width: 100%;
    height: 6px;
    background: #ddd;
    position: relative;
  }
  .conf-fill {
    height: 100%;
    width: 0;
    transition: width 0.6s ease, background 0.3s;
    background: var(--ink);
  }
  .verdict-box.real .conf-fill { background: var(--green); }
  .verdict-box.fake .conf-fill { background: var(--red); }

  /* METRICS SIDEBAR */
  .metrics {
    margin-top: 20px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .metric-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 14px;
    border: 1px solid #ddd;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
  }
  .metric-row .label { color: var(--muted); }
  .metric-row .val   { font-weight: 600; font-size: 1rem; }
  .metric-row .val.red   { color: var(--red); }
  .metric-row .val.green { color: var(--green); }

  /* MODEL INFO */
  .model-info {
    margin-top: 20px;
    padding: 14px;
    border: 1px dashed #bbb;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    line-height: 1.8;
  }
  .model-info strong { color: var(--ink); }

  /* HISTORY */
  .history-panel { grid-column: 1 / -1; margin-top: 8px; }
  .history-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 12px;
  }
  .history-card {
    border: 1px solid #ccc;
    padding: 14px 16px;
    font-size: 0.82rem;
    line-height: 1.5;
    position: relative;
    animation: fadeIn 0.3s ease;
  }
  @keyframes fadeIn { from { opacity:0; transform: translateY(6px); } to { opacity:1; transform:none; } }
  .history-card .hc-verdict {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.1rem;
    letter-spacing: 0.08em;
    position: absolute;
    top: 12px; right: 14px;
  }
  .history-card .hc-verdict.real { color: var(--green); }
  .history-card .hc-verdict.fake { color: var(--red); }
  .history-card .hc-text {
    color: var(--ink);
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    font-style: italic;
    margin-bottom: 6px;
    padding-right: 60px;
  }
  .history-card .hc-conf {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    color: var(--muted);
  }

  /* NOT TRAINED BANNER */
  .banner {
    padding: 12px 18px;
    background: var(--yellow);
    border: 2px solid var(--ink);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    margin-bottom: 20px;
    display: flex;
    gap: 10px;
    align-items: flex-start;
    line-height: 1.5;
  }
  .banner .icon { font-size: 1.1rem; flex-shrink: 0; }

  @media (max-width: 720px) {
    main { grid-template-columns: 1fr; }
    header { padding: 14px 16px; }
    header h1 { font-size: 2.2rem; }
  }
</style>
</head>
<body>

<header>
  <h1>FakeWatch</h1>
  <span class="edition">NLP Edition</span>
  <div class="status-dot">
    <span class="dot" id="modelDot"></span>
    <span id="modelStatus">checking model…</span>
  </div>
</header>

<main>
  <!-- LEFT: INPUT -->
  <div class="input-panel">
    <div class="section-label">Paste news article or headline</div>

    <div id="banner" class="banner" style="display:none">
      <span class="icon">⚠</span>
      <span>Model not trained yet. Click <strong>"Train Model"</strong> first, then come back to classify articles.</span>
    </div>

    <textarea id="articleText" placeholder="Paste your news article here…&#10;&#10;e.g. Scientists discover that eating chocolate daily improves memory by 400% according to a study funded by a chocolate company…"></textarea>
    <div class="char-count"><span id="charCount">0</span> characters</div>

    <div style="margin:14px 0 4px">
      <span class="section-label" style="border:none; margin:0; padding:0">Try samples →</span>
    </div>
    <div class="samples">
      <button class="sample-btn" onclick="loadSample('fake1')">🔴 Fake — Health Claim</button>
      <button class="sample-btn" onclick="loadSample('fake2')">🔴 Fake — Political</button>
      <button class="sample-btn" onclick="loadSample('real1')">🟢 Real — Science</button>
      <button class="sample-btn" onclick="loadSample('real2')">🟢 Real — Finance</button>
    </div>

    <button class="analyse-btn" id="analyseBtn" onclick="classify()">
      Analyse Article
      <span class="spinner"></span>
    </button>
  </div>

  <!-- RIGHT: VERDICT -->
  <div class="verdict-panel">
    <div class="section-label">Verdict</div>

    <div class="verdict-box waiting" id="verdictBox">
      <div class="verdict-stamp" id="verdictStamp">—</div>
      <div class="verdict-sub" id="verdictSub">Awaiting analysis</div>
      <div class="conf-wrap" id="confWrap" style="display:none">
        <div class="conf-label">
          <span>Confidence</span>
          <span id="confPct">0%</span>
        </div>
        <div class="conf-track"><div class="conf-fill" id="confFill"></div></div>
      </div>
    </div>

    <div class="metrics" id="metricsPanel">
      <div class="metric-row">
        <span class="label">FAKE PROBABILITY</span>
        <span class="val" id="mFake">—</span>
      </div>
      <div class="metric-row">
        <span class="label">REAL PROBABILITY</span>
        <span class="val" id="mReal">—</span>
      </div>
      <div class="metric-row">
        <span class="label">WORD COUNT</span>
        <span class="val" id="mWords">—</span>
      </div>
      <div class="metric-row">
        <span class="label">MODEL TYPE</span>
        <span class="val" id="mModel">—</span>
      </div>
    </div>

    <div class="model-info" id="modelInfo">
      <strong>Model Status</strong><br>
      Checking if trained model exists…
    </div>
  </div>

  <!-- HISTORY -->
  <div class="history-panel" id="historyPanel" style="display:none">
    <div class="section-label">Analysis History</div>
    <div class="history-grid" id="historyGrid"></div>
  </div>
</main>

<script>
const SAMPLES = {
  fake1: "DOCTORS HATE HIM! Local man discovers ancient trick that CURES cancer overnight using simple kitchen spice. Big Pharma is desperately trying to suppress this information. Share before it gets deleted! Scientists at an underground lab have confirmed this 100% natural remedy works on ALL types of cancer with no side effects.",
  fake2: "BREAKING: Secret documents LEAKED prove that the government is putting mind-control chemicals in tap water to keep the population docile. Insiders reveal a 30-year conspiracy involving top officials. The mainstream media refuses to cover this EXPLOSIVE truth. Wake up sheeple before it's too late!!!",
  real1: "Researchers at MIT have published findings in Nature demonstrating that a new machine learning approach can predict protein folding structures with 92% accuracy. The study, peer-reviewed by an independent panel, involved analysis of over 40,000 protein sequences and marks a significant step forward in computational biology.",
  real2: "The Federal Reserve announced Wednesday it will hold interest rates steady at 4.25–4.5 percent, citing continued uncertainty around trade policy and its potential effects on inflation. Fed Chair stated that the committee remains data-dependent and will monitor upcoming employment and price stability reports before any further adjustments."
};

let history = [];

function loadSample(key) {
  document.getElementById('articleText').value = SAMPLES[key];
  updateChar();
}

document.getElementById('articleText').addEventListener('input', updateChar);
function updateChar() {
  document.getElementById('charCount').textContent =
    document.getElementById('articleText').value.length;
}

async function checkModel() {
  try {
    const res = await fetch('/model_status');
    const d = await res.json();
    const dot = document.getElementById('modelDot');
    const status = document.getElementById('modelStatus');
    const info = document.getElementById('modelInfo');
    const banner = document.getElementById('banner');

    if (d.trained) {
      dot.style.background = '#1a6b3c';
      status.textContent = 'Model ready';
      banner.style.display = 'none';
      info.innerHTML = `<strong>Model Ready ✓</strong><br>
        Type: ${d.model_type}<br>
        Training accuracy: ${d.train_accuracy}<br>
        Test accuracy: ${d.test_accuracy}<br>
        Precision: ${d.precision} &nbsp;|&nbsp; Recall: ${d.recall}<br>
        F1 Score: ${d.f1}<br>
        Vocabulary size: ${d.vocab_size} tokens`;
      document.getElementById('mModel').textContent = d.model_type;
    } else {
      dot.style.background = '#e8c84a';
      status.textContent = 'Model not trained';
      banner.style.display = 'flex';
      info.innerHTML = `<strong>Model Not Trained</strong><br>
        Run <code>python train.py</code> to train the model first.<br>
        This will generate <code>model.pkl</code> and <code>vectorizer.pkl</code>.`;
    }
  } catch(e) {
    document.getElementById('modelStatus').textContent = 'Server offline';
  }
}

async function classify() {
  const text = document.getElementById('articleText').value.trim();
  if (!text) { alert('Please paste a news article first.'); return; }
  if (text.length < 20) { alert('Article too short. Please paste more text.'); return; }

  const btn = document.getElementById('analyseBtn');
  btn.classList.add('loading');
  btn.querySelector('span:first-child').style.opacity = '0.5';

  try {
    const res = await fetch('/classify', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text})
    });
    const d = await res.json();

    if (d.error) { alert('Error: ' + d.error); return; }

    showVerdict(d, text);
    addHistory(d, text);
  } catch(e) {
    alert('Could not connect to server. Is app.py running?');
  } finally {
    btn.classList.remove('loading');
    btn.querySelector('span:first-child').style.opacity = '1';
  }
}

function showVerdict(d, text) {
  const box = document.getElementById('verdictBox');
  const stamp = document.getElementById('verdictStamp');
  const sub = document.getElementById('verdictSub');
  const confWrap = document.getElementById('confWrap');
  const confFill = document.getElementById('confFill');
  const confPct = document.getElementById('confPct');

  box.className = 'verdict-box ' + d.label.toLowerCase();
  stamp.textContent = d.label.toUpperCase();
  sub.textContent = d.label === 'FAKE' ? '⚠ Likely misinformation' : '✓ Appears credible';

  confWrap.style.display = 'block';
  confPct.textContent = d.confidence + '%';

  setTimeout(() => {
    box.classList.add('shown');
    confFill.style.width = d.confidence + '%';
  }, 50);

  document.getElementById('mFake').textContent = d.fake_prob + '%';
  document.getElementById('mFake').className = 'val ' + (parseFloat(d.fake_prob) > 50 ? 'red' : '');
  document.getElementById('mReal').textContent = d.real_prob + '%';
  document.getElementById('mReal').className = 'val ' + (parseFloat(d.real_prob) > 50 ? 'green' : '');
  document.getElementById('mWords').textContent = text.split(/\s+/).filter(Boolean).length;
}

function addHistory(d, text) {
  history.unshift({label: d.label, confidence: d.confidence, text});
  if (history.length > 6) history.pop();

  const panel = document.getElementById('historyPanel');
  const grid = document.getElementById('historyGrid');
  panel.style.display = 'block';
  grid.innerHTML = history.map(h => `
    <div class="history-card">
      <div class="hc-verdict ${h.label.toLowerCase()}">${h.label}</div>
      <div class="hc-text">"${h.text}"</div>
      <div class="hc-conf">Confidence: ${h.confidence}%</div>
    </div>
  `).join('');
}

// Boot
checkModel();
</script>
</body>
</html>
"""

# ─── MODEL LOADING ─────────────────────────────────────────────────────────────
model = None
vectorizer = None
model_meta = {}

def load_model():
    global model, vectorizer, model_meta
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        with open("model_meta.json", "r") as f:
            model_meta = json.load(f)
        return True
    except FileNotFoundError:
        return False

# ─── ROUTES ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/model_status")
def model_status():
    trained = load_model()
    if trained:
        return jsonify({"trained": True, **model_meta})
    return jsonify({"trained": False})

@app.route("/classify", methods=["POST"])
def classify():
    if not load_model():
        return jsonify({"error": "Model not trained. Run: python train.py"}), 400

    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Preprocess & vectorize
    cleaned = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    vec = vectorizer.transform([cleaned])
    proba = model.predict_proba(vec)[0]

    # Classes: 0 = FAKE, 1 = REAL (based on training labels)
    fake_prob = round(proba[0] * 100, 1)
    real_prob = round(proba[1] * 100, 1)
    label = "REAL" if real_prob > fake_prob else "FAKE"
    confidence = max(fake_prob, real_prob)

    return jsonify({
        "label": label,
        "confidence": confidence,
        "fake_prob": fake_prob,
        "real_prob": real_prob
    })

if __name__ == "__main__":
    print("\n🔍 FakeWatch — Fake News Detector")
    print("=" * 40)
    if os.path.exists("model.pkl"):
        print("✅ Model found! Starting server...")
    else:
        print("⚠️  Model not trained yet. Run: python train.py")
        print("   Then restart app.py")
    print("\n🌐 Open: http://127.0.0.1:5000")
    print("=" * 40 + "\n")
    app.run(debug=True, port=5000)
