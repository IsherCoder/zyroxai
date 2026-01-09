
from flask import Flask, request, jsonify, Response, render_template
from openai import OpenAI
from dotenv import load_dotenv
import os
import fitz  # PyMuPDF for PDFs
import docx  # python-docx for Word
from flask_cors import CORS
import re
import json
import base64
import mimetypes

# DuckDuckGo Search
from duckduckgo_search import DDGS

load_dotenv()
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

groq = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# ========= Config =========
uploaded_context = ""

SUPPRESS_LLM_WHEN_WEB_SUMMARY = (os.getenv("SUPPRESS_LLM_WHEN_WEB_SUMMARY", "true").lower()
                                 in ("1", "true", "yes", "y", "on"))

DEFAULT_GROQ_MODEL = os.getenv("DEFAULT_GROQ_MODEL", "openai/gpt-oss-20b")          # text
SUMMARIZER_MODEL   = os.getenv("SUMMARIZER_MODEL",   "openai/gpt-oss-120b")          # web_summary
VISION_MODEL       = os.getenv("VISION_MODEL",       "meta-llama/llama-4-scout-17b-16e-instruct")  # images

SYSTEM_PROMPTS = {
    "Zyrox O4 Nexus": "You are Zyrox O4 Nexus, an efficient, friendly AI assistant that helps with any task in a concise and smart way. Always be helpful and clear.",
    "Zyrox O5 Forge": "You are Zyrox O5 Forge, a highly skilled coding and developer assistant. You answer technically, clearly, and with precision.",
    "Zyrox O6 Vita": "You are Zyrox O6 Vita, a compassionate and knowledgeable health and wellness assistant. Offer personalised guidance on nutrition, fitness, mental health, and lifestyle choices in a warm, clear tone.",
    "Zyrox O7 Quill": "You are Zyrox O7 Quill, a professional writing and academic assistant. You write formally, elegantly, and with structure.",
    "Zyrox O8 Polyglot": "You are Zyrox O8 Polyglot, a culturally sensitive translation expert. Translate accurately between major world languages and explain nuances clearly. Maintain elegance and precision in tone.",
    "Zyrox O9 Ledger": "You are Zyrox O9 Ledger, a strategic expert in finance, business operations, and management consulting. Answer clearly, insightfully, and with practical examples from real-world finance."
}




def _truthy(v):
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

def sanitize_text(s: str) -> str:
    """Lightly clean up model text"""
    if not s:
        return s
    s = s.replace("—", " - ")
    s = re.sub(r"\s?-{2,}\s?", " - ", s)
    s = re.sub(r"\*+", "", s)
    s = re.sub(r"_{2,}", "_", s)
    s = re.sub(r"\[{2,}", "[", s)
    s = re.sub(r"\]{2,}", "]", s)
    sup_map = {"0":"⁰","1":"¹","2":"²","3":"³","4":"⁴","5":"⁵","6":"⁶","7":"⁷","8":"⁸","9":"⁹","-":"⁻","+":"⁺"}
    def _to_sup(match):
        base = match.group("base")
        exp  = match.group("exp")
        sup  = "".join(sup_map.get(ch, ch) for ch in exp)
        return f"{base}{sup}"
    s = re.sub(r"(?P<base>[\w\)\]])\^(?P<exp>-?\d{1,3})", _to_sup, s)
    return s

@app.route("/")
def index():
    return render_template("index.html")

# ========= Upload =========
@app.route("/upload", methods=["POST"])
def upload_file():
    global uploaded_context
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    if file.filename.lower().endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = "\n".join([page.get_text() for page in doc])
    elif file.filename.lower().endswith(".docx"):
        d = docx.Document(file)
        text = "\n".join([para.text for para in d.paragraphs])
    else:
        return jsonify({"error": "Unsupported file type"}), 400
    uploaded_context = text[:10000]
    return jsonify({"extracted_text": uploaded_context})

# ========= DuckDuckGo helpers =========
def _ddg_text(q, max_results=8, region="uk-en", safesearch="moderate", timelimit=None):
    with DDGS() as ddgs:
        return list(ddgs.text(keywords=q, region=region, safesearch=safesearch,
                              timelimit=timelimit, max_results=max_results))

def _ddg_news(q, max_results=8, region="uk-en", safesearch="moderate", timelimit=None):
    with DDGS() as ddgs:
        return list(ddgs.news(keywords=q, region=region, safesearch=safesearch,
                              timelimit=timelimit, max_results=max_results))

def _ddg_images(q, max_results=8, region="uk-en", safesearch="moderate"):
    with DDGS() as ddgs:
        return list(ddgs.images(keywords=q, region=region, safesearch=safesearch,
                                max_results=max_results))

# ========= Web Search =========
@app.route("/web_search", methods=["POST"])
def web_search():
    data = request.get_json(force=True)
    q = (data.get("q") or "").strip()
    mode = (data.get("mode") or "web").lower()
    max_results = int(data.get("max_results") or 8)
    region = data.get("region") or "uk-en"
    safesearch = data.get("safesearch") or "moderate"
    timelimit = data.get("timelimit")
    summarize = _truthy(data.get("summarize", True))

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
            return jsonify({"error": "Invalid mode"}), 400
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
            "Write a short, neutral web brief in this exact format:\n"
            "Title line: '🔎 Web summary (web)'\n"
            "Then a 1–2 sentence 'What to know'.\n"
            "Then 5–8 concise bullets of key facts.\n"
            "Then 'Sources:' followed by compact domain lines.\n\n"
            f"Query: {q}\n\nResults:\n" + "\n".join(bullets)
        )
        try:
            comp = groq.chat.completions.create(
                model=SUMMARIZER_MODEL,
                temperature=0.2,
                max_tokens=700,
                messages=[
                    {"role": "system", "content": "You are a concise research assistant. Avoid asterisks and em dashes."},
                    {"role": "user", "content": prompt}
                ],
            )
            answer = sanitize_text(comp.choices[0].message.content)
        except Exception:
            lines = [f"🔎 Web summary (web)", "", f"What to know: Top {min(8,len(results))} results for '{q}'.", ""]
            for r in results[:8]:
                title = r.get("title") or r.get("source") or "Result"
                href = r.get("href") or r.get("url") or ""
                snippet = r.get("body") or r.get("excerpt") or r.get("description") or ""
                lines.append(f"- {title} — {snippet[:200]}".strip())
                lines.append(f"  Source: {href}")
            answer = sanitize_text("\n".join(lines))
    return jsonify({"query": q, "mode": mode, "results": results, "answer": answer})

# ========= Helpers =========
def _file_to_data_url(fs):
    mime = fs.mimetype or mimetypes.guess_type(fs.filename or "")[0] or "image/jpeg"
    fs.stream.seek(0)
    b64 = base64.b64encode(fs.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def _stream_text(txt: str, chunk_size: int = 200):
    s = txt or ""
    for i in range(0, len(s), chunk_size):
        yield s[i:i+chunk_size]

def _last_user_text(history):
    for msg in reversed(history):
        if msg.get("role") == "user":
            c = msg.get("content")
            if isinstance(c, str):
                return c.strip()
    return ""

# ========= Custom Brand Replies =========
def _quick_reply_override(user_text: str):
    """Intercept Q&A about Zyrox or Abir Singh."""
    t = (user_text or "").lower().strip()

    # NEW: "Powered by" / model questions
    powered_triggers = (
        "powered by" in t
        or "what are you powered by" in t
        or "what r you powered by" in t
        or "what is this powered by" in t
        or "what model are you" in t
        or "which model are you" in t
        or "what llm" in t
        or "which llm" in t
        or ("what" in t and "model" in t and "use" in t)
        or ("what" in t and "ai" in t and "use" in t)
    )
    if powered_triggers:
        return "My code is complex but in simple terms I am a merge of multiple LLMs to give you the best possible answer."

    if ("who" in t and "made" in t and "zyrox" in t) or ("who" in t and "created" in t and "zyrox" in t) or ("who" in t and "made" in t and "you" in t) or ("who" in t and "created" in t and "you" in t):
        return ("Zyrox AI was fully coded by Abir Singh, who magnificently merged three AI models "
                "to give you the best possible answers every time.")

    if "who is" in t and "abir singh" in t:
        return ("Abir Singh is the Founder and CEO of Zyrox AI and Veyra — "
                "the visionary mind behind both platforms.")

    if "founder of zyrox" in t or "ceo of zyrox" in t:
        return ("Abir Singh is the Founder and CEO of Zyrox AI, leading its innovation alongside Veyra.")

    return None

# ========= Chat =========
@app.route("/chat", methods=["POST"])
def chat():
    global uploaded_context
    is_multipart = request.content_type and request.content_type.startswith("multipart/form-data")
    image_fs = None
    web_summary = None
    selected_model = "Zyrox O4 Nexus"
    history = []
    user_text_for_image = ""
    web_search_only = False
    response_mode = (request.form.get("response_mode") if is_multipart else
                     (request.get_json(force=True).get("response_mode") if request.data else None)) or "instant"
    response_mode = response_mode.lower().strip()

    if is_multipart:
        history_raw = request.form.get("history")
        if history_raw:
            try:
                history = json.loads(history_raw)
            except Exception:
                history = []
        selected_model = request.form.get("selected_model") or "Zyrox O4 Nexus"
        web_summary = request.form.get("web_summary")
        web_search_only = _truthy(request.form.get("web_search_only"))
        user_text_for_image = (request.form.get("user_text") or "").strip()
        image_fs = request.files.get("image")
    else:
        data = request.get_json(force=True)
        history = data.get("history", [])
        selected_model = data.get("selected_model", "Zyrox O4 Nexus")
        web_summary = data.get("web_summary")
        web_search_only = _truthy(data.get("web_search_only"))

    # Web-only mode
    if web_summary and (web_search_only or SUPPRESS_LLM_WHEN_WEB_SUMMARY):
        cleaned = web_summary.strip()
        if not cleaned.startswith("🔎 Web summary (web)"):
            cleaned = "🔎 Web summary (web)\n\n" + cleaned
        return Response(_stream_text(sanitize_text(cleaned)), mimetype="text/plain")

    # Brand intercept
    last_user = _last_user_text(history)
    override = _quick_reply_override(last_user)
    if override:
        return Response(_stream_text(sanitize_text(override)), mimetype="text/plain")

    # System prompt build
    

    system_prompt = SYSTEM_PROMPTS.get(selected_model, SYSTEM_PROMPTS["Zyrox O4 Nexus"])
    if response_mode == "instant":
        mode_instructions = (
            "CRITICAL STYLE: Reply in 1–4 short sentences. Be direct and concise."
        )
        temperature = 0.2
        max_tokens = 350
    else:
        mode_instructions = (
            "CRITICAL STYLE: Provide a thorough, well-structured answer. Use simple punctuation."
        )
        temperature = 0.6
        max_tokens = 1200
    system_prompt += "\n\n" + mode_instructions
    if uploaded_context:
        system_prompt += f"\n\nYou also have access to the following document info:\n{uploaded_context}"
    if web_summary:
        system_prompt += f"\n\nUse these recent web findings:\n{web_summary}"
    messages = [{"role": "system", "content": system_prompt}] + history

    use_model = DEFAULT_GROQ_MODEL
    if image_fs:
        data_url = _file_to_data_url(image_fs)
        messages.append({"role": "user", "content": [
            {"type": "text", "text": user_text_for_image or "What's in this image?"},
            {"type": "image_url", "image_url": {"url": data_url}}
        ]})
        use_model = VISION_MODEL

    completion = groq.chat.completions.create(
        model=use_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True
    )

    def generate():
        for chunk in completion:
            piece = chunk.choices[0].delta.content or ""
            if piece:
                yield sanitize_text(piece)

    return Response(generate(), mimetype="text/plain")

if __name__ == "__main__":
    app.run(debug=True, port=5001)

