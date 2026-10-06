"""
FastAPI app serving a scikit-learn text-classification Pipeline
(TfidfVectorizer + LogisticRegression) saved as model.pkl.

Run locally:
    uvicorn main:app --reload

Open:
    http://127.0.0.1:8000/        -> web UI
    http://127.0.0.1:8000/docs    -> Swagger docs

Test with curl:
    curl -X POST http://127.0.0.1:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"text": "I loved this movie"}'
"""

import logging
import os

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# ---------- Logging (shows up in the Render "Logs" tab) ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("ml-api")

# ---------- App + model ----------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

app = FastAPI(
    title="Getting Started with ML in Production API",
    description="Text classification API: model -> FastAPI -> Docker -> Render.",
    version="1.0.0",
)

try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded from %s", MODEL_PATH)
except Exception as e:  # missing file, version mismatch, etc.
    model = None
    logger.error("Could not load model: %s", e)


# ---------- Schemas ----------
class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to classify")


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float | None = None


# ---------- Routes ----------
@app.get("/health")
def health():
    """Health check for Render / load balancers."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        logger.error("Prediction requested but model is not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded.")

    # The Pipeline contains the vectorizer, so we pass raw text in a list.
    pred = model.predict([request.text])[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = round(float(max(model.predict_proba([request.text])[0])), 4)

    logger.info("Input: %r -> Prediction: %s (confidence=%s)", request.text[:200], pred, confidence)
    return PredictionResponse(prediction=str(pred), confidence=confidence)


# ---------- Simple web UI (served at /) ----------
PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Text Classifier</title>
<style>
  :root { --bg:#f6f7fb; --card:#fff; --ink:#1c2333; --muted:#5d667a; --line:#dde1ec; --brand:#2f5bea; --ok:#0f8a5f; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#12151f; --card:#1b2030; --ink:#eef0f7; --muted:#9aa3b8; --line:#2b3248; --brand:#7b9bff; --ok:#3ddc97; }
  }
  * { box-sizing: border-box; }
  body { margin:0; min-height:100vh; display:grid; place-items:center; padding:20px;
         background:var(--bg); color:var(--ink); font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { width:100%; max-width:560px; background:var(--card); border:1px solid var(--line);
         border-radius:14px; padding:28px; }
  h1 { margin:0 0 6px; font-size:1.5rem; }
  p.sub { margin:0 0 20px; color:var(--muted); }
  textarea { width:100%; min-height:130px; padding:12px; font:inherit; color:var(--ink);
             background:var(--bg); border:1px solid var(--line); border-radius:10px; resize:vertical; }
  textarea:focus, button:focus-visible { outline:2px solid var(--brand); outline-offset:2px; }
  .row { display:flex; gap:10px; margin-top:12px; flex-wrap:wrap; }
  button { padding:10px 18px; font:inherit; font-weight:600; border-radius:10px; cursor:pointer; border:1px solid var(--brand); }
  #go { background:var(--brand); color:#fff; }
  #clear { background:transparent; color:var(--brand); }
  button:disabled { opacity:.6; cursor:wait; }
  #result { margin-top:20px; padding:16px; border-radius:10px; border:1px solid var(--line); display:none; }
  #label { font-size:1.4rem; font-weight:700; color:var(--ok); }
  .bar { height:8px; background:var(--line); border-radius:99px; margin-top:10px; overflow:hidden; }
  .bar > div { height:100%; width:0; background:var(--ok); transition:width .4s; }
  .err { color:#d93a3a; }
  footer { margin-top:18px; font-size:.85rem; color:var(--muted); }
  a { color:var(--brand); }
</style>
</head>
<body>
<main>
  <h1>Text classifier</h1>
  <p class="sub">Type or paste some text and the model will predict its class.</p>
  <textarea id="text" placeholder="e.g. I loved this movie, the acting was brilliant"></textarea>
  <div class="row">
    <button id="go">Predict</button>
    <button id="clear" type="button">Clear</button>
  </div>
  <div id="result" aria-live="polite">
    <div>Prediction</div>
    <div id="label"></div>
    <div id="conf" class="sub" style="color:var(--muted)"></div>
    <div class="bar"><div id="fill"></div></div>
  </div>
  <footer>API docs: <a href="/docs">/docs</a> &middot; Health: <a href="/health">/health</a></footer>
</main>
<script>
  const $ = id => document.getElementById(id);
  $("go").onclick = async () => {
    const text = $("text").value.trim();
    if (!text) { $("text").focus(); return; }
    $("go").disabled = true; $("go").textContent = "Predicting...";
    const box = $("result"); box.style.display = "block";
    try {
      const r = await fetch("/predict", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({text})
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || "Request failed");
      $("label").className = ""; $("label").textContent = d.prediction;
      if (d.confidence != null) {
        const pct = (d.confidence * 100).toFixed(1);
        $("conf").textContent = "Confidence: " + pct + "%"; $("fill").style.width = pct + "%";
      } else { $("conf").textContent = ""; $("fill").style.width = "0"; }
    } catch (e) {
      $("label").className = "err"; $("label").textContent = "Error: " + e.message;
      $("conf").textContent = ""; $("fill").style.width = "0";
    }
    $("go").disabled = false; $("go").textContent = "Predict";
  };
  $("clear").onclick = () => { $("text").value = ""; $("result").style.display = "none"; $("text").focus(); };
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE