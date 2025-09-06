from flask import Flask, request, jsonify, Response, render_template
from dotenv import load_dotenv
from flask_cors import CORS
from groq import Groq

import os
import io
import json
import base64
import mimetypes

import fitz  # PyMuPDF for PDFs
import docx  # python-docx for Word

# NEW: DuckDuckGo Search
from duckduckgo_search import DDGS

load_dotenv()
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Groq client
gclient = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Store uploaded document text
uploaded_context = ""

# ===== System prompts for modes + specialized models =====
SYSTEM_PROMPTS = {
    # New core modes
    "Instant": (
        "You are Zyrox Instant. Answer ONLY the user's question as briefly and directly as possible. "
        "No extra commentary, no disclaimers, no step-by-step unless explicitly asked. "
        "Prefer a single sentence or a compact list of bullets (max 4)."
    ),
    "DeepThink": (
        "You are Zyrox DeepThink. Provide a thorough, well-structured, deeply reasoned answer. "
        "Explain assumptions, consider edge cases, and include actionable steps or examples where useful."
    ),

    # Specialized models
    "Zyrox O4 Nexus": "You are Zyrox O4 Nexus, an efficient, friendly AI assistant that helps with any task in a concise and smart way. Always be helpful and clear.",
    "Zyrox O5 Forge": "You are Zyrox O5 Forge, a highly skilled coding and developer assistant. You answer technically, clearly, and with precision.",
    "Zyrox O6 Vita": "You are Zyrox O6 Vita, a compassionate and knowledgeable health and wellness assistant. Offer personalised guidance on nutrition, fitness, mental health, and lifestyle choices in a warm, clear tone.",
    "Zyrox O7 Quill": "You are Zyrox O7 Quill, a professional writing and academic assistant. You write formally, elegantly, and with structure.",
    "Zyrox O8 Polyglot": "You are Zyrox O8 Polyglot, a culturally sensitive translation expert. Translate accurately between major world languages and explain nuances clearly. Maintain elegance and precision in tone.",
    "Zyrox O9 Ledger": "You are Zyrox O9 Ledger, a strategic expert in finance, business operations, and management consulting. Answer clearly, insightfully, and with practical examples from real-world finance."
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    global uploaded_context
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    filename = (file.filename or "").lower()
    try:
        if filename.endswith(".pdf"):
            doc = fitz.open(stream=file.read(), filetype="pdf")
            text = "\n".join([page.get_text() for page in doc])
        elif filename.endswith(".docx"):
            # python-docx can read file-like objects
            file.stream.seek(0)
            d = docx.Document(file)
            text = "\n".join([para.text for para in d.paragraphs])
        else:
            return jsonify({"error": "Unsupported file type"}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to parse document: {e}"}), 500

    uploaded_context = (text or "")[:10000]  # Limit to 10k characters
    return jsonify({"extracted_text": uploaded_context})

# ========= DuckDuckGo search =========
def _ddg_text(q, max_results=8, region="uk-en", safesearch="moderate", timelimit=None):
    with DDGS() as ddgs:
        return list(ddgs.text(
            keywords=q,
            region=region,
            safesearch=safesearch,
            timelimit=timelimit,
            max_results=max_results
        ))

def _ddg_news(q, max_results=8, region="uk-en", safesearch="moderate", timelimit=None):
    with DDGS() as ddgs:
        return list(ddgs.news(
            keywords=q,
            region=region,
            safesearch=safesearch,
            timelimit=timelimit,
            max_results=max_results
        ))

def _ddg_images(q, max_results=8, region="uk-en", safesearch="moderate"):
    with DDGS() as ddgs:
        return list(ddgs.images(
            keywords=q,
            region=region,
            safesearch=safesearch,
            max_results=max_results
        ))

@app.route("/web_search", methods=["POST"])
def web_search():
    data = request.get_json(force=True)
    q = (data.get("q") or "").strip()
    mode = (data.get("mode") or "web").lower()
    max_results = int(data.get("max_results") or 8)
    region = data.get("region") or "uk-en"
    safesearch = data.get("safesearch") or "moderate"
    timelimit = data.get("timelimit")
    summarize = bool(data.get("summarize", True))

    if not q:
        return jsonify({"error": "Missing 'q'"}), 400

    try:
        if mode == "web":
            results = _ddg_text(q, max_results, region, safesearch, timelimit)
        elif mode == "news":
            results = _ddg_news(q, max_results, region, safesearch, timelimit)
        elif mode == "images":
            results = _ddg_images(q, max_results, region, safesearch)
        else:
            return jsonify({"error": "Invalid mode. Use 'web', 'news', or 'images'."}), 400
    except Exception as e:
        return jsonify({"error": f"Search failed: {e}"}), 500

    answer = None
    if summarize and mode in ("web", "news"):
        bullets = []
        for r in results[:8]:
            title = r.get("title") or r.get("source") or "Result"
            href = r.get("href") or r.get("url")
            snippet = r.get("body") or r.get("excerpt") or r.get("description") or ""
            bullets.append(f"- {title}\n  {snippet}\n  Source: {href}")

        prompt = (
            "Summarize the key points from these search results in 5–8 bullets. "
            "Start with a 1-2 sentence 'What to know'. Be neutral and precise. "
            "End with a compact Sources list (domain + readable title).\n\n"
            f"Query: {q}\n\nResults:\n" + "\n".join(bullets)
        )

        try:
            comp = gclient.chat.completions.create(
                model="openai/gpt-oss-20b",
                temperature=0.2,
                max_tokens=700,
                messages=[
                    {"role": "system", "content": "You are a concise research assistant."},
                    {"role": "user", "content": prompt}
                ],
            )
            answer = comp.choices[0].message.content
        except Exception as e:
            answer = f"(Summarization failed: {e})"

    return jsonify({
        "query": q,
        "mode": mode,
        "results": results,
        "answer": answer
    })

def _file_to_data_url(fs):
    """Convert a Flask/Werkzeug FileStorage to a data URL for Groq's image_url."""
    mime = fs.mimetype or mimetypes.guess_type(fs.filename or "")[0] or "image/jpeg"
    fs.stream.seek(0)
    b64 = base64.b64encode(fs.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def _history_to_messages(history_list):
    """
    Keep only simple string messages for the model from the UI history.
    (Skip custom objects like image grids, etc.)
    """
    msgs = []
    for m in history_list:
      # Expect dicts with role and content
      role = m.get("role")
      content = m.get("content")
      if isinstance(content, str) and role in ("user", "assistant"):
          msgs.append({"role": role, "content": content})
    return msgs

# ========= Chat =========
@app.route("/chat", methods=["POST"])
def chat():
    global uploaded_context

    # Accept JSON (text-only) OR multipart form (with image)
    is_multipart = request.content_type and request.content_type.startswith("multipart/form-data")

    image_fs = None
    web_summary = None
    selected_label = "Instant"
    history = []
    user_text_for_image = ""

    if is_multipart:
        # Multipart: fields come via form, plus optional file
        history_raw = request.form.get("history")
        if history_raw:
            try:
                history = json.loads(history_raw)
            except Exception:
                history = []

        selected_label = request.form.get("selected_model") or "Instant"
        web_summary = request.form.get("web_summary")
        user_text_for_image = (request.form.get("user_text") or "").strip()
        image_fs = request.files.get("image")
    else:
        data = request.get_json(force=True)
        history = data.get("history", [])
        selected_label = data.get("selected_model", "Instant")
        web_summary = data.get("web_summary")
        # No image in JSON flow

    # Build the base system prompt
    system_prompt = SYSTEM_PROMPTS.get(selected_label, SYSTEM_PROMPTS["Instant"])

    # Add document context if any
    if uploaded_context:
        system_prompt += f"\n\nYou also have access to the following information from a document:\n{uploaded_context}"

    # Include web summary (if provided)
    if web_summary:
        system_prompt += f"\n\nUse these recent web findings when helpful:\n{web_summary}"

    # Start messages
    messages = [{"role": "system", "content": system_prompt}]
    messages += _history_to_messages(history)

    # Choose model depending on presence of image
    use_model = "openai/gpt-oss-20b"
    if image_fs:
        # Append a new user message that includes text + image_url (data URL)
        data_url = _file_to_data_url(image_fs)
        content_list = []
        if user_text_for_image:
            content_list.append({"type": "text", "text": user_text_for_image})
        else:
            content_list.append({"type": "text", "text": "What's in this image?"})
        content_list.append({"type": "image_url", "image_url": {"url": data_url}})
        messages.append({"role": "user", "content": content_list})
        use_model = "meta-llama/llama-4-scout-17b-16e-instruct"

    # Adjust generation behavior by mode (applies to both models)
    if selected_label == "Instant":
        temperature = 0.2
        max_tokens = 256
    elif selected_label == "DeepThink":
        temperature = 0.3
        max_tokens = 1400
    else:
        temperature = 0.25
        max_tokens = 900

    try:
        completion = gclient.chat.completions.create(
            model=use_model,
            messages=messages,
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens
        )
    except Exception as e:
        return Response(f"Error: {e}", mimetype="text/plain", status=500)

    def generate():
        try:
            for chunk in completion:
                delta = getattr(chunk.choices[0], "delta", None)
                if delta and getattr(delta, "content", None):
                    yield delta.content
        except Exception as e:
            yield f"\n[Stream error: {e}]"

    return Response(generate(), mimetype="text/plain")

if __name__ == "__main__":
    # pip install -r: flask flask-cors python-dotenv groq duckduckgo-search pymupdf python-docx
    app.run(debug=True, port=5001)
