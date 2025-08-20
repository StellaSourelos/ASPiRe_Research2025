from flask import Flask, request, render_template_string, redirect
import sqlite3
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
import torch
import os
from collections import Counter
import pandas as pd
import warnings
from sentence_transformers import SentenceTransformer, util
from datetime import datetime
import math
from statistics import mean

warnings.filterwarnings("ignore", message="resource_tracker: There appear to be .* leaked semaphore")
torch.set_num_threads(1)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

app = Flask(__name__)

# === GPT-2 SETUP ===
tokenizer = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=True)
model = GPT2LMHeadModel.from_pretrained("gpt2", local_files_only=True)
model.to(device)
model.eval()

# === EMBEDDING SETUP ===
embedder = SentenceTransformer("all-MiniLM-L6-v2")

LEADERBOARD_PATH = "leaderboard_log.jsonl"
FEEDBACK_PATH = "feedback_log.json"
EVAL_PATH = "evaluation_results.jsonl"

# === SIMPLE BLEU IMPLEMENTATION (BLEU-1 & BLEU-2) ===
def _tokenize(text):
    return [t for t in text.strip().lower().split() if t]

def _ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def _modified_precision(candidate_tokens, reference_tokens, n):
    cand_ngrams = Counter(_ngrams(candidate_tokens, n))
    ref_ngrams = Counter(_ngrams(reference_tokens, n))
    overlap = 0
    total = sum(cand_ngrams.values())
    if total == 0:
        return 0.0
    for ng, count in cand_ngrams.items():
        overlap += min(count, ref_ngrams.get(ng, 0))
    return overlap / total

def _brevity_penalty(c_len, r_len):
    if c_len == 0:
        return 0.0
    if c_len > r_len:
        return 1.0
    return math.exp(1 - (r_len / max(c_len, 1)))

def bleu_scores(candidate, reference):
    cand_t = _tokenize(candidate)
    ref_t  = _tokenize(reference)
    if not ref_t or not cand_t:
        return 0.0, 0.0
    p1 = _modified_precision(cand_t, ref_t, 1)
    p2 = _modified_precision(cand_t, ref_t, 2)
    bp = _brevity_penalty(len(cand_t), len(ref_t))
    bleu1 = bp * (p1 if p1 > 0 else 0.0)
    if p1 == 0 or p2 == 0:
        bleu2 = 0.0
    else:
        bleu2 = bp * math.exp(0.5 * (math.log(p1) + math.log(p2)))
    return bleu1, bleu2

# === SEMANTIC SIMILARITY (answer vs retrieved context) ===
def semantic_similarity(answer, reference):
    if not answer.strip() or not reference.strip():
        return 0.0
    try:
        a = embedder.encode([answer], convert_to_tensor=True)
        b = embedder.encode([reference], convert_to_tensor=True)
        sim = util.pytorch_cos_sim(a, b).item()
        return max(0.0, min(1.0, float(sim)))
    except Exception:
        return 0.0

# === SEMANTIC RETRIEVAL ===
def retrieve(query, top_n=3):
    conn = sqlite3.connect("knowledge.db")
    df = pd.read_sql("SELECT id, content FROM knowledge", conn)
    conn.close()

    if df.empty:
        return "", []

    # Compute embeddings for contents and query
    try:
        content_embeddings = embedder.encode(df["content"].tolist(), convert_to_tensor=True)
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(query_embedding, content_embeddings)[0]
    except Exception:
        # If embedding fails, fall back to naive ranking
        scores = [0.0] * len(df)

    # Load feedback-based doc scores
    doc_scores = Counter()
    if os.path.exists(FEEDBACK_PATH):
        with open(FEEDBACK_PATH, "r") as f:
            for line in f:
                log = json.loads(line)
                score = 1 if log.get("thumbs") == "up" else -1
                for doc_id in log.get("docs_used", []):
                    doc_scores[doc_id] += score

    adjusted_scores = []
    for i in range(len(df)):
        base = float(scores[i]) if not isinstance(scores, list) else float(scores[i])
        adjusted_scores.append((i, base + 0.1 * doc_scores.get(int(df['id'].iloc[i]), 0)))

    top_results = sorted(adjusted_scores, key=lambda x: x[1], reverse=True)[:top_n]

    context = " ".join(df["content"].iloc[i] for i, _ in top_results)
    doc_ids = [int(df["id"].iloc[i]) for i, _ in top_results]
    return context, doc_ids

# === GPT GENERATION ===
def generate(query, context):
    try:
        if context.strip():
            prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
        else:
            prompt = f"Question: {query}\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model.generate(
            **inputs,
            max_length=150,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded.split("Answer:")[-1].strip()
    except Exception as e:
        print("Error in generate():", e)
        return "Sorry, generation failed."

# === SAVE/LOAD HELPERS ===
def save_feedback(log):
    with open(FEEDBACK_PATH, "a") as f:
        f.write(json.dumps(log) + "\n")

def save_leaderboard(entry):
    with open(LEADERBOARD_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

def load_leaderboard():
    rows = []
    if os.path.exists(LEADERBOARD_PATH):
        with open(LEADERBOARD_PATH, "r") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows

def append_eval_row(row):
    with open(EVAL_PATH, "a") as f:
        f.write(json.dumps(row) + "\n")

def load_eval_results():
    rows = []
    if os.path.exists(EVAL_PATH):
        with open(EVAL_PATH, "r") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows

# === TEST QUERIES (~50) ===
TEST_QUERIES = [
    "What is a microcontroller?",
    "Explain supervised vs unsupervised learning.",
    "What is an embedded system?",
    "Define convolution in neural networks.",
    "What is overfitting in machine learning?",
    "How does backpropagation work?",
    "What is transfer learning?",
    "Explain attention in transformers.",
    "What is a database index?",
    "Explain normalization in databases.",
    "What is ACID in databases?",
    "What is the bias-variance tradeoff?",
    "Explain gradient descent.",
    "What is L2 regularization?",
    "Define precision and recall.",
    "What is cosine similarity?",
    "Explain TF-IDF.",
    "What is a knowledge graph?",
    "What is vector search?",
    "What is semantic search?",
    "Explain RAG (retrieval augmented generation).",
    "What is tokenization?",
    "What is lemmatization?",
    "What is stemming?",
    "Explain cross entropy loss.",
    "What is dropout?",
    "Explain batch normalization.",
    "What is a hash table?",
    "Explain sorting algorithms.",
    "Define time complexity.",
    "What is Big-O notation?",
    "Explain dynamic programming.",
    "What is memoization?",
    "What is multi-processing vs multi-threading?",
    "Explain REST vs GraphQL.",
    "What is a message queue?",
    "What is Kafka?",
    "Explain microservices.",
    "What is containerization?",
    "Explain Docker.",
    "What is Kubernetes?",
    "Explain CI/CD.",
    "What is unit testing?",
    "What is integration testing?",
    "Explain A/B testing.",
    "What is feature engineering?",
    "What is dimensionality reduction?",
    "Explain PCA.",
    "What is k-means clustering?",
    "Explain hierarchical clustering."
]

# === HTML TEMPLATE (main page) ===
TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>RAG Prototype</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 40px; background-color: #f9f9f9; color: #333; }
    h2 { color: #2c3e50; }
    form { margin-bottom: 20px; background-color: #fff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1); }
    input[type="text"] { width: 100%; padding: 10px; font-size: 16px; margin-top: 8px; margin-bottom: 12px; border: 1px solid #ccc; border-radius: 5px; }
    textarea { width: 100%; padding: 10px; font-size: 14px; border-radius: 5px; border: 1px solid #ccc; }
    input[type="submit"], button { padding: 10px 20px; font-size: 14px; border: none; border-radius: 5px; background-color: #3498db; color: white; cursor: pointer; margin-top: 10px; }
    input[type="submit"]:hover, button:hover { background-color: #2980b9; }
    table { width: 100%; border-collapse: collapse; background: #fff; }
    th, td { padding: 8px 12px; border: 1px solid #ccc; text-align: left; }
    .thumbs button { font-size: 20px; margin-right: 10px; }
    a { color: #2980b9; text-decoration: none; }
    a:hover { text-decoration: underline; }
    .card { background:#fff; padding:16px; border-radius:10px; box-shadow: 0 2px 8px rgba(0,0,0,0.07); margin-top: 16px; }
    .badge { display:inline-block; padding:4px 8px; border-radius:999px; background:#eef5ff; color:#245; font-size:12px; margin-right:6px; }
  </style>
</head>
<body>
<div style="max-width: 1000px; margin: auto;">
  <h2>RAG Prototype (SQLite Knowledge Base)</h2>
  <form action="/" method="post">
    <label>Enter your question:</label><br>
    <input type="text" name="query" value="{{ query or '' }}"><br>
    <input type="submit" value="Submit">
  </form>

  <div class="card">
    <b>Utilities:</b>
    <a href="/leaderboard">Leaderboard</a> •
    <a href="/dashboard">📊 Feedback Dashboard</a> •
    <a href="/evaluation">🧪 Batch Evaluation</a>
    <form action="/evaluation/run" method="post" style="display:inline; margin-left: 10px;">
        <button type="submit">Run Batch Evaluation</button>
    </form>
  </div>

  {% if query %}
    <div class="card">
      <h3>Generated Answer</h3>
      <p>{{ answer }}</p>
      <div>
        <span class="badge">BLEU-1: {{ bleu1_pct }}%</span>
        <span class="badge">BLEU-2: {{ bleu2_pct }}%</span>
        <span class="badge">Semantic: {{ semantic_pct }}%</span>
        <span class="badge"><b>Overall: {{ overall_pct }}%</b></span>
      </div>
    </div>

    <form class="card" action="/feedback" method="post">
      <h3>Feedback</h3>
      <input type="hidden" name="query" value="{{ query }}">
      <input type="hidden" name="answer" value="{{ answer }}">
      <input type="hidden" name="docs" value="{{ docs }}">
      <div class="thumbs">
        <button name="thumbs" value="up">👍</button>
        <button name="thumbs" value="down">👎</button>
      </div>
      <br>
      <label>Edit the response (optional):</label><br>
      <textarea name="edit" rows="4"></textarea><br>
      <input type="submit" value="Submit Feedback">
    </form>

    <div class="card">
      <h3>🏆 Leaderboard (Top 10 by Overall Score)</h3>
      {% if leaderboard and leaderboard|length > 0 %}
      <table>
        <tr>
          <th>#</th>
          <th>When</th>
          <th>Query</th>
          <th>BLEU-1</th>
          <th>BLEU-2</th>
          <th>Semantic</th>
          <th>Overall</th>
        </tr>
        {% for row in leaderboard %}
        <tr>
          <td>{{ loop.index }}</td>
          <td>{{ row["timestamp"] }}</td>
          <td style="max-width:300px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{{ row["query"] }}</td>
          <td>{{ "%.1f"|format(100*row["bleu1"]) }}%</td>
          <td>{{ "%.1f"|format(100*row["bleu2"]) }}%</td>
          <td>{{ "%.1f"|format(100*row["semantic"]) }}%</td>
          <td><b>{{ "%.1f"|format(100*row["overall"]) }}%</b></td>
        </tr>
        {% endfor %}
      </table>
      {% else %}
        <p>No leaderboard data yet.</p>
      {% endif %}
    </div>
  {% endif %}
</div>
</body>
</html>
"""

# === EVALUATION PAGE TEMPLATE ===
EVAL_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>RAG Batch Evaluation</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 40px; background-color: #f9f9f9; color: #333; }
    h2 { color: #2c3e50; }
    .card { background:#fff; padding:16px; border-radius:10px; box-shadow:0 2px 8px rgba(0,0,0,0.07); margin-top:16px; }
    table { width: 100%; border-collapse: collapse; background:#fff; }
    th, td { padding: 8px 12px; border: 1px solid #ccc; text-align: left; }
    a { color:#2980b9; text-decoration:none; }
    a:hover { text-decoration:underline; }
    button { padding: 10px 20px; background:#3498db; color:#fff; border:none; border-radius:6px; cursor:pointer; }
    button:hover { background:#2980b9; }
    .badge { display:inline-block; padding:4px 8px; border-radius:999px; background:#eef5ff; color:#245; font-size:12px; margin-right:6px; }
  </style>
</head>
<body>
  <div style="max-width: 1100px; margin:auto;">
    <h2>🧪 Batch Evaluation (RAG vs No-RAG)</h2>
    <div class="card">
      <form action="/evaluation/run" method="post" style="display:inline;">
        <button type="submit">Run Evaluation (~50 queries)</button>
      </form>
      <a style="margin-left:10px" href="/evaluation/clear">Clear Results</a> •
      <a style="margin-left:10px" href="/">Back to App</a>
    </div>

    {% if summary %}
    <div class="card">
      <h3>Summary</h3>
      <div>
        <span class="badge">Runs: {{ summary["n"] }}</span>
        <span class="badge">Avg BLEU-1 (RAG): {{ "%.1f"|format(100*summary["rag_bleu1_avg"]) }}%</span>
        <span class="badge">Avg BLEU-1 (No-RAG): {{ "%.1f"|format(100*summary["no_bleu1_avg"]) }}%</span>
        <span class="badge">Avg BLEU-2 (RAG): {{ "%.1f"|format(100*summary["rag_bleu2_avg"]) }}%</span>
        <span class="badge">Avg BLEU-2 (No-RAG): {{ "%.1f"|format(100*summary["no_bleu2_avg"]) }}%</span>
        <span class="badge">Avg Semantic (RAG): {{ "%.1f"|format(100*summary["rag_sem_avg"]) }}%</span>
        <span class="badge">Avg Semantic (No-RAG): {{ "%.1f"|format(100*summary["no_sem_avg"]) }}%</span>
        <span class="badge"><b>Avg Overall (RAG): {{ "%.1f"|format(100*summary["rag_overall_avg"]) }}%</b></span>
        <span class="badge"><b>Avg Overall (No-RAG): {{ "%.1f"|format(100*summary["no_overall_avg"]) }}%</b></span>
      </div>
    </div>
    {% endif %}

    {% if rows and rows|length > 0 %}
    <div class="card">
      <h3>Detailed Results</h3>
      <table>
        <tr>
          <th>#</th>
          <th>When</th>
          <th>Query</th>
          <th>BLEU-1 (RAG)</th>
          <th>BLEU-1 (No-RAG)</th>
          <th>BLEU-2 (RAG)</th>
          <th>BLEU-2 (No-RAG)</th>
          <th>Semantic (RAG)</th>
          <th>Semantic (No-RAG)</th>
          <th>Overall (RAG)</th>
          <th>Overall (No-RAG)</th>
        </tr>
        {% for r in rows %}
        <tr>
          <td>{{ loop.index }}</td>
          <td>{{ r["timestamp"] }}</td>
          <td style="max-width:300px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{{ r["query"] }}</td>
          <td>{{ "%.1f"|format(100*r["rag_bleu1"]) }}%</td>
          <td>{{ "%.1f"|format(100*r["no_bleu1"]) }}%</td>
          <td>{{ "%.1f"|format(100*r["rag_bleu2"]) }}%</td>
          <td>{{ "%.1f"|format(100*r["no_bleu2"]) }}%</td>
          <td>{{ "%.1f"|format(100*r["rag_sem"]) }}%</td>
          <td>{{ "%.1f"|format(100*r["no_sem"]) }}%</td>
          <td><b>{{ "%.1f"|format(100*r["rag_overall"]) }}%</b></td>
          <td><b>{{ "%.1f"|format(100*r["no_overall"]) }}%</b></td>
        </tr>
        {% endfor %}
      </table>
    </div>
    {% else %}
      <div class="card"><p>No evaluation results yet. Click “Run Evaluation”.</p></div>
    {% endif %}
  </div>
</body>
</html>
"""

# === ROUTES ===
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form["query"]
        if not query.strip():
            return "<p><b>Error:</b> Empty query provided.</p><a href='/'>Go back</a>"

        # Retrieve and generate
        context, used_docs = retrieve(query)
        answer = generate(query, context)

        # Scoring (use retrieved context as the reference; if none, fall back to query)
        ref_for_bleu = context if context.strip() else query
        bleu1, bleu2 = bleu_scores(answer, ref_for_bleu)
        semantic = semantic_similarity(answer, context if context else "")
        # Overall score: simple blend (50% semantic, 50% BLEU-2). Tweak as needed.
        overall = 0.5 * semantic + 0.5 * bleu2

        # Save to leaderboard
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "answer": answer,
            "bleu1": float(bleu1),
            "bleu2": float(bleu2),
            "semantic": float(semantic),
            "overall": float(overall),
            "docs_used": used_docs
        }
        save_leaderboard(entry)

        # Build Top 10 leaderboard
        rows = load_leaderboard()
        rows_sorted = sorted(rows, key=lambda r: r.get("overall", 0.0), reverse=True)[:10]

        if not used_docs:
            answer += "\n\n(Note: No relevant context found; this answer is purely generated from the question.)"

        return render_template_string(
            TEMPLATE,
            query=query,
            answer=answer,
            docs=json.dumps(used_docs),
            bleu1_pct=f"{100*bleu1:.1f}",
            bleu2_pct=f"{100*bleu2:.1f}",
            semantic_pct=f"{100*semantic:.1f}",
            overall_pct=f"{100*overall:.1f}",
            leaderboard=rows_sorted
        )

    # GET
    return render_template_string(TEMPLATE)

@app.route("/feedback", methods=["POST"])
def feedback():
    query = request.form.get("query", "")
    answer = request.form.get("answer", "")
    docs_used_str = request.form.get("docs", "[]")
    thumbs = request.form.get("thumbs", "")
    edited = request.form.get("edit", "")

    try:
        docs_used = json.loads(docs_used_str)
    except Exception:
        docs_used = []

    save_feedback({
        "query": query,
        "answer": answer,
        "thumbs": thumbs,
        "docs_used": docs_used,
        "edited": edited
    })

    return f"<p>Thanks for the feedback! <a href='/'>Ask another</a> or <a href='/dashboard'>View Dashboard</a></p>"

@app.route("/dashboard")
def dashboard():
    updated = request.args.get("updated")
    confirmation_html = ""
    if updated == "1":
        confirmation_html = "<p style='color: green; font-weight: bold;'>Document updated successfully!</p>"

    try:
        logs = []
        if os.path.exists(FEEDBACK_PATH):
            with open(FEEDBACK_PATH, "r") as f:
                for line in f:
                    logs.append(json.loads(line))

        if not logs:
            return "<h3>No feedback data yet.</h3><a href='/'>Back</a>"

        conn = sqlite3.connect("knowledge.db")
        df = pd.read_sql("SELECT * FROM knowledge", conn)
        conn.close()

        rows_html = "".join(
            f"""
            <tr>
                <form action='/update_doc' method='post'>
                    <td><input type='hidden' name='doc_id' value='{row['id']}'>{row['id']}</td>
                    <td><textarea name='content' rows='3' cols='70'>{row['content']}</textarea></td>
                    <td><input type='submit' value='Update'></td>
                </form>
            </tr>
            """ for _, row in df.iterrows()
        )

        total_feedback = len(logs)
        thumbs_up = sum(1 for log in logs if log.get("thumbs") == "up")
        thumbs_down = total_feedback - thumbs_up

        doc_usage = Counter()
        for log in logs:
            doc_usage.update(log.get("docs_used", []))

        df_feedback = pd.DataFrame(logs)
        feedback_table = df_feedback.to_html(classes="data", index=False)

        return f"""
        {confirmation_html}
        <h2>📊 Feedback Dashboard</h2>
        <p>Total Feedback: {total_feedback}</p>
        <p>👍 Thumbs Up: {thumbs_up}</p>
        <p>👎 Thumbs Down: {thumbs_down}</p>

        <h3>Document Usage Frequency:</h3>
        <ul>
            {''.join(f"<li>Doc ID {doc_id}: {count} uses</li>" for doc_id, count in doc_usage.items())}
        </ul>

        <h3>Full Feedback Log:</h3>
        {feedback_table}

        <h3>Edit Knowledge Base:</h3>
        <table border='1'>
            <tr><th>ID</th><th>Content</th><th>Action</th></tr>
            {rows_html}
        </table>

        <br><a href='/'>⬅ Back to app</a>
        """
    except Exception as e:
        return f"<p>Error loading dashboard: {e}</p><a href='/'>Back</a>"

@app.route("/leaderboard")
def leaderboard():
    rows = load_leaderboard()
    if not rows:
        return "<h3>No leaderboard data yet.</h3><a href='/'>Back</a>"
    rows_sorted = sorted(rows, key=lambda r: r.get("overall", 0.0), reverse=True)
    table_rows = "".join(
        f"""
        <tr>
          <td>{i+1}</td>
          <td>{row.get("timestamp","")}</td>
          <td style='max-width:300px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;'>{row.get("query","")}</td>
          <td>{row.get("answer","")[:120]}{"..." if len(row.get("answer",""))>120 else ""}</td>
          <td>{100*row.get("bleu1",0.0):.1f}%</td>
          <td>{100*row.get("bleu2",0.0):.1f}%</td>
          <td>{100*row.get("semantic",0.0):.1f}%</td>
          <td><b>{100*row.get("overall",0.0):.1f}%</b></td>
        </tr>
        """ for i, row in enumerate(rows_sorted)
    )
    return f"""
    <h2>🏆 Leaderboard</h2>
    <table border="1" cellpadding="6" cellspacing="0">
      <tr>
        <th>#</th><th>When</th><th>Query</th><th>Answer (preview)</th><th>BLEU-1</th><th>BLEU-2</th><th>Semantic</th><th>Overall</th>
      </tr>
      {table_rows}
    </table>
    <br><a href='/'>⬅ Back to app</a>
    """

@app.route("/update_doc", methods=["POST"])
def update_doc():
    doc_id = request.form.get("doc_id")
    content = request.form.get("content")

    if not doc_id or content is None:
        return "<p>Invalid data.</p><a href='/dashboard'>Back to dashboard</a>"

    try:
        conn = sqlite3.connect("knowledge.db")
        cursor = conn.cursor()
        cursor.execute("UPDATE knowledge SET content = ? WHERE id = ?", (content, doc_id))
        conn.commit()
        conn.close()
        return redirect("/dashboard?updated=1")
    except Exception as e:
        return f"<p>Error updating document: {e}</p><a href='/dashboard'>Back</a>"

# === BATCH EVALUATION HELPERS ===
def evaluate_query_pair(query):
    """
    For a single query:
      - Retrieve context (for reference only)
      - Generate answer with RAG (with context)
      - Generate answer without RAG (empty context)
      - Score both vs the reference (context if available else query)
    """
    context, used_docs = retrieve(query)
    # Reference for both systems:
    reference = context if context.strip() else query

    # RAG answer (uses context)
    rag_answer = generate(query, context)
    rag_bleu1, rag_bleu2 = bleu_scores(rag_answer, reference)
    rag_sem = semantic_similarity(rag_answer, context if context else "")
    rag_overall = 0.5 * rag_sem + 0.5 * rag_bleu2

    # No-RAG answer (no context)
    no_answer = generate(query, "")
    no_bleu1, no_bleu2 = bleu_scores(no_answer, reference)
    no_sem = semantic_similarity(no_answer, context if context else "")
    no_overall = 0.5 * no_sem + 0.5 * no_bleu2

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "reference_used": "context" if context.strip() else "query",
        "rag_bleu1": float(rag_bleu1),
        "rag_bleu2": float(rag_bleu2),
        "rag_sem": float(rag_sem),
        "rag_overall": float(rag_overall),
        "no_bleu1": float(no_bleu1),
        "no_bleu2": float(no_bleu2),
        "no_sem": float(no_sem),
        "no_overall": float(no_overall),
    }
    return row

def eval_summary(rows):
    if not rows:
        return None
    return {
        "n": len(rows),
        "rag_bleu1_avg": mean(r["rag_bleu1"] for r in rows),
        "no_bleu1_avg": mean(r["no_bleu1"] for r in rows),
        "rag_bleu2_avg": mean(r["rag_bleu2"] for r in rows),
        "no_bleu2_avg": mean(r["no_bleu2"] for r in rows),
        "rag_sem_avg": mean(r["rag_sem"] for r in rows),
        "no_sem_avg": mean(r["no_sem"] for r in rows),
        "rag_overall_avg": mean(r["rag_overall"] for r in rows),
        "no_overall_avg": mean(r["no_overall"] for r in rows),
    }

# === BATCH EVALUATION ROUTES ===
@app.route("/evaluation", methods=["GET"])
def evaluation_page():
    rows = load_eval_results()
    summary = eval_summary(rows) if rows else None
    return render_template_string(EVAL_TEMPLATE, rows=rows, summary=summary)

@app.route("/evaluation/run", methods=["POST"])
def evaluation_run():
    # Run evaluation over TEST_QUERIES; append results
    for q in TEST_QUERIES:
        row = evaluate_query_pair(q)
        append_eval_row(row)
    # Redirect to page
    return redirect("/evaluation")

@app.route("/evaluation/clear")
def evaluation_clear():
    try:
        if os.path.exists(EVAL_PATH):
            os.remove(EVAL_PATH)
    except Exception:
        pass
    return redirect("/evaluation")

# === MAIN ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
