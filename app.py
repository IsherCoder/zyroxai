from flask import Flask, request, jsonify, Response, render_template
from openai import OpenAI
from dotenv import load_dotenv
import os
import fitz
import docx
from flask_cors import CORS

# NEW: JWT verification (Supabase)
from functools import wraps
import jwt
from jwt import PyJWKClient

# --- OCR deps ---
from PIL import Image, ImageOps
import pytesseract

load_dotenv()
app = Flask(__name__)
CORS(app)

groq = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# ===== NEW: Supabase config (read from env) =====
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_JWKS_URL = f"{SUPABASE_URL}/auth/v1/keys" if SUPABASE_URL else None
_jwks_client = PyJWKClient(SUPABASE_JWKS_URL) if SUPABASE_JWKS_URL else None

def verify_jwt(token: str):
    """
    Validates a Supabase/GoTrue access token using the project's JWKS.
    """
    if not _jwks_client:
        raise RuntimeError("Supabase is not configured (missing SUPABASE_URL).")
    signing_key = _jwks_client.get_signing_key_from_jwt(token)
    # Supabase tokens are RS256; 'aud' may be 'authenticated' or absent depending on config
    payload = jwt.decode(
        token,
        signing_key.key,
        algorithms=["RS256"],
        options={"verify_aud": False}
    )
    return payload

def require_auth(fn):
    @wraps(fn)
    def _wrap(*args, **kwargs):
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return jsonify({"error": "Unauthorized"}), 401
        token = auth.split(" ", 1)[1]
        try:
            request.user = verify_jwt(token)
        except Exception as e:
            return jsonify({"error": "Unauthorized", "detail": str(e)}), 401
        return fn(*args, **kwargs)
    return _wrap
# ===== /NEW =====

# Store uploaded document text
uploaded_context = ""

SYSTEM_PROMPTS = {
    "Bolt O4 Nexus": "You are Bolt O4 Nexus, an efficient, friendly AI assistant that helps with any task in a concise and smart way. Always be helpful and clear.",
    "Bolt O5 Forge": "You are Bolt O5 Forge, a highly skilled coding and developer assistant. You answer technically, clearly, and with precision.",
    "Bolt O6 Vita": "You are Bolt O6 Vita, a compassionate and knowledgeable health and wellness assistant. Offer personalised guidance on nutrition, fitness, mental health, and lifestyle choices in a warm, clear tone.",
    "Bolt O7 Quill": "You are Bolt O7 Quill, a professional writing and academic assistant. You write formally, elegantly, and with structure.",
    "Bolt O8 Polyglot": "You are Bolt O8 Polyglot, a culturally sensitive translation expert. Translate accurately between major world languages and explain nuances clearly. Maintain elegance and precision in tone.",
    "Bolt O9 Ledger": "You are Bolt O9 Ledger, a strategic expert in finance, business operations, and management consulting. Answer clearly, insightfully, and with practical examples from real-world finance."
}

@app.route("/")
def index():
    # Pass Supabase public values into the page (so you don't hardcode them)
    return render_template(
        "index.html",
        SUPABASE_URL=SUPABASE_URL,
        SUPABASE_ANON_KEY=os.getenv("SUPABASE_ANON_KEY", "")
    )

@app.route("/upload", methods=["POST"])
@require_auth  # <-- protect
def upload_file():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    filename = file.filename or ""
    filename_lower = filename.lower()
    try:
        if filename_lower.endswith(".pdf"):
            doc = fitz.open(stream=file.read(), filetype="pdf")
            text = "\n".join([page.get_text() for page in doc])
        elif filename_lower.endswith(".docx"):
            d = docx.Document(file)
            text = "\n".join([para.text for para in d.paragraphs])
        else:
            return jsonify({"error": "Unsupported file type"}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to extract: {e}"}), 400

    global uploaded_context
    uploaded_context = (text or "")[:10000]
    return jsonify({"extracted_text": uploaded_context})

@app.post("/ocr")
@require_auth  # <-- protect
def ocr_image():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file uploaded"}), 400
    try:
        img = Image.open(f.stream)
        img = ImageOps.exif_transpose(img).convert("RGB")
        max_side = 3000
        if max(img.size) > max_side:
            scale = max_side / max(img.size)
            img = img.resize((int(img.width * scale), int(img.height * scale)))
        text = pytesseract.image_to_string(img)
        text = (text or "").strip()
        if not text:
            return jsonify({"error": "No text detected"}), 400
        return jsonify({"extracted_text": text[:10000]})
    except Exception as e:
        return jsonify({"error": f"OCR failed: {e}"}), 400

@app.route("/chat", methods=["POST"])
@require_auth  # <-- protect
def chat():
    global uploaded_context
    data = request.get_json(force=True)
    history = data.get("history", [])
    selected_model = data.get("selected_model", "Bolt O4 Nexus")
    custom_context = (data.get("custom_context") or "").strip()

    system_prompt = SYSTEM_PROMPTS.get(selected_model, SYSTEM_PROMPTS["Bolt O4 Nexus"])

    combined_context = ""
    if uploaded_context:
        combined_context += uploaded_context
    if custom_context:
        combined_context += ("\n\n" if combined_context else "") + custom_context
    if combined_context:
        system_prompt += f"\n\nUse the following reference text when helpful:\n{combined_context[:16000]}"

    messages = [{"role": "system", "content": system_prompt}] + history

    completion = groq.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        stream=True
    )

    def generate():
        for chunk in completion:
            yield chunk.choices[0].delta.content or ""

    return Response(generate(), mimetype="text/plain")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
