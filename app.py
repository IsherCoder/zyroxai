from flask import Flask, request, jsonify, Response, render_template
from openai import OpenAI
from dotenv import load_dotenv
import os
import fitz  # PyMuPDF for PDFs
import docx  # python-docx for Word
from flask_cors import CORS

# NEW: DuckDuckGo Search
from duckduckgo_search import DDGS

load_dotenv()
app = Flask(__name__, static_folder="static", template_folder="templates")

CORS(app)

groq = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# Store uploaded document text
uploaded_context = ""

SYSTEM_PROMPTS = {
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

    if file.filename.lower().endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = "\n".join([page.get_text() for page in doc])
    elif file.filename.lower().endswith(".docx"):
        d = docx.Document(file)
        text = "\n".join([para.text for para in d.paragraphs])
    else:
        return jsonify({"error": "Unsupported file type"}), 400

    uploaded_context = text[:10000]  # Limit to 10k characters
    return jsonify({"extracted_text": uploaded_context})

# ========= NEW: Web Search Endpoint (DuckDuckGo) =========
def _ddg_text(q, max_results=8, region="uk-en", safesearch="moderate", timelimit=None):
    with DDGS() as ddgs:
        return list(ddgs.text(
            keywords=q,
            region=region,
            safesearch=safesearch,      # "off" | "moderate" | "strict"
            timelimit=timelimit,        # "d" | "w" | "m" | "y" or None
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
    timelimit = data.get("timelimit")  # None | "d" | "w" | "m" | "y"
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
            comp = groq.chat.completions.create(
                model="llama3-70b-8192",
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

# ========= Chat (unchanged) =========
@app.route("/chat", methods=["POST"])
def chat():
    global uploaded_context
    data = request.get_json()
    history = data.get("history", [])
    selected_model = data.get("selected_model", "Zyrox O4 Nexus")

    system_prompt = SYSTEM_PROMPTS.get(selected_model, SYSTEM_PROMPTS["Zyrox O4 Nexus"])

    # Add document context
    if uploaded_context:
        system_prompt += f"\n\nYou also have access to the following information from a document:\n{uploaded_context}"

    # Optionally include a web_summary (if frontend sent one)
    web_summary = data.get("web_summary")
    if web_summary:
        system_prompt += f"\n\nUse these recent web findings when helpful:\n{web_summary}"

    messages = [{"role": "system", "content": system_prompt}] + history

    completion = groq.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        stream=True
    )

    def generate():
        for chunk in completion:
            yield chunk.choices[0].delta.content or ""

    return Response(generate(), mimetype="text/plain")

if __name__ == "__main__":
    app.run(debug=True, port=5001)
