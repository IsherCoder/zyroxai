import os
import re
import json
import base64
import mimetypes
import datetime
from urllib.parse import urlparse

from flask import Flask, request, jsonify, Response, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI

import fitz  # PyMuPDF for PDFs
import docx  # python-docx for Word

import httpx
from bs4 import BeautifulSoup

# DuckDuckGo Search
from duckduckgo_search import DDGS

load_dotenv()

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Groq OpenAI-compatible client
groq = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# ========= Config =========
uploaded_context = ""

SUPPRESS_LLM_WHEN_WEB_SUMMARY = (os.getenv("SUPPRESS_LLM_WHEN_WEB_SUMMARY", "true").lower()
                                 in ("1", "true", "yes", "y", "on"))

DEFAULT_GROQ_MODEL = os.getenv("DEFAULT_GROQ_MODEL", "openai/gpt-oss-20b")          # chat text
SUMMARIZER_MODEL   = os.getenv("SUMMARIZER_MODEL",   "openai/gpt-oss-120b")         # web_summary
VISION_MODEL       = os.getenv("VISION_MODEL",       "meta-llama/llama-4-scout-17b-16e-instruct")  # images

SYSTEM_PROMPTS = {
    "Zyrox O4 Nexus": "You are Zyrox O4 Nexus, an efficient, friendly AI assistant that helps with any task in a concise and smart way. Always be helpful and clear.",
    "Zyrox O5 Forge": "You are Zyrox O5 Forge, a highly skilled coding and developer assistant. You answer technically, clearly, and with precision.",
    "Zyrox O6 Vita": "You are Zyrox O6 Vita, a compassionate and knowledgeable health and wellness assistant. Offer personalised guidance on nutrition, fitness, mental health, and lifestyle choices in a warm, clear tone.",
    "Zyrox O7 Quill": "You are Zyrox O7 Quill, a professional writing and academic assistant. You write formally, elegantly, and with structure.",
    "Zyrox O8 Polyglot": "You are Zyrox O8 Polyglot, a culturally sensitive translation expert. Translate accurately between major world languages and explain nuances clearly. Maintain elegance and precision in tone.",
    "Zyrox O9 Ledger": "You are Zyrox O9 Ledger, a strategic expert in finance, business operations, and management consulting. Answer clearly, insightfully, and with practical examples from real-world finance."
}

IDENTITY_AND_PROVENANCE_RULES = """
Identity and provenance rules:
- You are the Zyrox AI assistant inside the Zyrox app.
- Do not mention OpenAI, training data, model providers, or internal tooling unless the user explicitly asks.
- If asked who made Zyrox, say it was built by Abir Singh and the Zyrox team.
- If asked what powers Zyrox, say it uses third-party language models accessed via an API provider.
- Never claim you were created by OpenAI or ChatGPT.
"""

# ========= Utilities =========
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

    # Superscript conversion for simple exponents like n^2, x^-1, 10^3
    sup_map = {"0":"⁰","1":"¹","2":"²","3":"³","4":"⁴","5":"⁵","6":"⁶","7":"⁷","8":"⁸","9":"⁹","-":"⁻","+":"⁺"}
    def _to_sup(match):
        base = match.group("base")
        exp  = match.group("exp")
        sup  = "".join(sup_map.get(ch, ch) for ch in exp)
        return f"{base}{sup}"
    s = re.sub(r"(?P<base>[\w\)\]])\^(?P<exp>-?\d{1,3})", _to_sup, s)
    return s

def _stream_text(txt: str, chunk_size: int = 200):
    s = txt or ""
    for i in range(0, len(s), chunk_size):
        yield s[i:i+chunk_size]

def _file_to_data_url(fs):
    mime = fs.mimetype or mimetypes.guess_type(fs.filename or "")[0] or "image/jpeg"
    fs.stream.seek(0)
    b64 = base64.b64encode(fs.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def _last_user_text(history):
    for msg in reversed(history):
        if msg.get("role") == "user":
            c = msg.get("content")
            if isinstance(c, str):
                return c.strip()
    return ""

def _domain(u: str) -> str:
    try:
        host = urlparse(u).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""

def _looks_like_weather_query(q: str) -> bool:
    t = (q or "").lower()
    triggers = [
        "weather", "forecast", "temperature", "rain", "snow", "wind",
        "met office", "humidity", "uv index", "storm", "sunrise", "sunset"
    ]
    return any(k in t for k in triggers)

def _fetch_page_text(url: str, timeout_s: int = 10, max_chars: int = 12000) -> str:
    """
    Fetch HTML and extract readable text.
    Conservative to avoid timeouts and huge pages on Render.
    """
    if not url:
        return ""

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; Milo/ZyroxWebFetcher)"
    }

    try:
        with httpx.Client(follow_redirects=True, timeout=timeout_s, headers=headers) as client:
            r = client.get(url)
            if r.status_code != 200 or not r.text:
                return ""

        soup = BeautifulSoup(r.text, "lxml")

        # Remove noisy elements
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]):
            tag.decompose()

        # Prefer <article> if present
        article = soup.find("article")
        text_source = article.get_text(" ", strip=True) if article else soup.get_text(" ", strip=True)

        # Compress whitespace and clamp size
        text_source = re.sub(r"\s+", " ", text_source).strip()
        return text_source[:max_chars]
    except Exception:
        return ""

# ========= Routes =========
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
def _ddg_text(q, max_results=10, region="uk-en", safesearch="moderate", timelimit=None):
    with DDGS() as ddgs:
        return list(ddgs.text(keywords=q, region=region, safesearch=safesearch,
                              timelimit=timelimit, max_results=max_results))

def _ddg_news(q, max_results=10, region="uk-en", safesearch="moderate", timelimit=None):
    with DDGS() as ddgs:
        return list(ddgs.news(keywords=q, region=region, safesearch=safesearch,
                              timelimit=timelimit, max_results=max_results))

def _ddg_images(q, max_results=10, region="uk-en", safesearch="moderate"):
    with DDGS() as ddgs:
        return list(ddgs.images(keywords=q, region=region, safesearch=safesearch,
                                max_results=max_results))

# ========= Web Search (improved: fetch pages + summarise) =========
@app.route("/web_search", methods=["POST"])
def web_search():
    data = request.get_json(force=True) or {}
    q = (data.get("q") or "").strip()
    mode = (data.get("mode") or "web").lower()
    max_results = int(data.get("max_results") or 10)
    region = data.get("region") or "uk-en"
    safesearch = data.get("safesearch") or "moderate"
    timelimit = data.get("timelimit")
    summarize = _truthy(data.get("summarize", True))

    # Hard rule: no weather fetching in this build
    if _looks_like_weather_query(q):
        return jsonify({
            "query": q,
            "mode": mode,
            "results": [],
            "answer": "Weather requests are disabled in this build. Please ask a non-weather question."
        }), 200

    if not q:
        return jsonify({"error": "Missing 'q'"}), 400

    # 1) Search DDG
    try:
        if mode == "web":
            results = _ddg_text(q, max_results=max_results, region=region, safesearch=safesearch, timelimit=timelimit)
        elif mode == "news":
            results = _ddg_news(q, max_results=max_results, region=region, safesearch=safesearch, timelimit=timelimit)
        elif mode == "images":
            results = _ddg_images(q, max_results=max_results, region=region, safesearch=safesearch)
        else:
            return jsonify({"error": "Invalid mode"}), 400
    except Exception as e:
        return jsonify({"error": f"Search failed: {e}"}), 500

    # Images: no summary
    if mode == "images":
        return jsonify({"query": q, "mode": mode, "results": results, "answer": None})

    # No summary requested: return raw results
    if not summarize:
        return jsonify({"query": q, "mode": mode, "results": results, "answer": None})

    # 2) Fetch top pages and extract text
    MAX_PAGES = int(os.getenv("WEB_FETCH_PAGES", "6"))
    pages = []

    for r in results:
        url = r.get("href") or r.get("url") or ""
        title = r.get("title") or r.get("source") or "Result"
        snippet = r.get("body") or r.get("excerpt") or r.get("description") or ""
        if not url:
            continue

        text = _fetch_page_text(url)
        if text:
            pages.append({
                "title": title,
                "url": url,
                "domain": _domain(url),
                "snippet": snippet,
                "content": text
            })

        if len(pages) >= MAX_PAGES:
            break

    today = datetime.date.today().isoformat()

    # Fallback if page fetch failed: summarise snippets only
    if not pages:
        lines = [f"Web summary", f"What to know: Could not fetch page text; summarised snippets only (as of {today}).", ""]
        for r in results[:8]:
            title = r.get("title") or r.get("source") or "Result"
            url = r.get("href") or r.get("url") or ""
            body = r.get("body") or r.get("excerpt") or r.get("description") or ""
            lines.append(f"- {title}: {body[:220]}".strip())
            if url:
                lines.append(f"  Source: {url}")
        return jsonify({"query": q, "mode": mode, "results": results, "answer": sanitize_text("\n".join(lines))}), 200

    # 3) Summarise using Groq with today's date included
    source_blocks = []
    for i, p in enumerate(pages, start=1):
        source_blocks.append(
            f"[{i}] {p['title']}\nURL: {p['url']}\nDomain: {p['domain']}\nExtract: {p['content']}\n"
        )

    prompt = (
        f"As of {today}, summarise the topic using only the provided sources.\n\n"
        "Output rules:\n"
        "- Start with: 'Web summary'\n"
        "- Then 1–2 sentences: 'What to know'\n"
        "- Then 6–10 bullet points of key facts (flat list)\n"
        "- Then 'Sources:' with one line per domain\n"
        "- Do not mention internal tools.\n\n"
        f"Query: {q}\n\n"
        "Sources:\n" + "\n\n".join(source_blocks)
    )

    try:
        comp = groq.chat.completions.create(
            model=SUMMARIZER_MODEL,
            temperature=0.2,
            max_tokens=900,
            messages=[
                {"role": "system", "content": "You are a neutral research summariser. Be factual. Avoid emojis and em dashes."},
                {"role": "user", "content": prompt}
            ],
        )
        answer = sanitize_text(comp.choices[0].message.content)
        domains = sorted({p["domain"] for p in pages if p["domain"]})
        return jsonify({"query": q, "mode": mode, "results": results, "answer": answer, "domains": domains}), 200
    except Exception as e:
        return jsonify({"error": f"Summariser failed: {e}"}), 500

# ========= Custom Brand Replies =========
def _quick_reply_override(user_text: str):
    """
    Intercept Q&A about Zyrox identity/power.
    Keeps it vendor-neutral and avoids provider mentions unless explicitly asked.
    """
    t = (user_text or "").lower().strip()

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
        return "Zyrox uses third-party language models accessed via an API provider."

    made_triggers = (
        ("who" in t and "made" in t and ("zyrox" in t or "you" in t))
        or ("who" in t and "created" in t and ("zyrox" in t or "you" in t))
        or ("who built" in t and ("zyrox" in t or "you" in t))
    )
    if made_triggers:
        return "Zyrox was built by Abir Singh and the Zyrox team."

    if "who is" in t and "abir" in t and "singh" in t:
        return "Abir Singh is the Founder and CEO of Zyrox."

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

    response_mode = (
        request.form.get("response_mode") if is_multipart else
        (request.get_json(force=True).get("response_mode") if request.data else None)
    ) or "instant"
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
        data = request.get_json(force=True) or {}
        history = data.get("history", [])
        selected_model = data.get("selected_model", "Zyrox O4 Nexus")
        web_summary = data.get("web_summary")
        web_search_only = _truthy(data.get("web_search_only"))

    # Web-only mode (optional)
    if web_summary and (web_search_only or SUPPRESS_LLM_WHEN_WEB_SUMMARY):
        cleaned = web_summary.strip()
        return Response(_stream_text(sanitize_text(cleaned)), mimetype="text/plain")

    # Brand intercept
    last_user = _last_user_text(history)
    override = _quick_reply_override(last_user)
    if override:
        return Response(_stream_text(sanitize_text(override)), mimetype="text/plain")

    # System prompt build
    system_prompt = (
        SYSTEM_PROMPTS.get(selected_model, SYSTEM_PROMPTS["Zyrox O4 Nexus"])
        + "\n\n"
        + IDENTITY_AND_PROVENANCE_RULES
    )

    if response_mode == "instant":
        mode_instructions = "CRITICAL STYLE: Reply in 1–4 short sentences. Be direct and concise."
        temperature = 0.2
        max_tokens = 350
    else:
        mode_instructions = "CRITICAL STYLE: Provide a thorough, well-structured answer. Use simple punctuation."
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
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_text_for_image or "What's in this image?"},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]
        })
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
    # Render uses PORT env var
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
