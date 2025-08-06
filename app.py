from flask import Flask, request, jsonify, Response
from flask import Flask, request, jsonify, Response, render_template
from openai import OpenAI
from dotenv import load_dotenv
import os
import fitz  # PyMuPDF for PDFs
import docx  # python-docx for Word
from flask_cors import CORS

load_dotenv()
app = Flask(__name__)
CORS(app)

groq = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# Store uploaded document text
uploaded_context = ""

SYSTEM_PROMPTS = {
    "ISHER O4 Nexus": "You are Isher O4 Nexus, an efficient, friendly AI assistant that helps with any task in a concise and smart way. Always be helpful and clear.",
    "ISHER O5 Forge": "You are Isher O5 Forge, a highly skilled coding and developer assistant. You answer technically, clearly, and with precision.",
    "ISHER O6 Vita": "You are Isher O6 Vita, a compassionate and knowledgeable health and wellness assistant. Offer personalised guidance on nutrition, fitness, mental health, and lifestyle choices in a warm, clear tone.",
    "ISHER O7 Quill": "You are Isher O7 Quill, a professional writing and academic assistant. You write formally, elegantly, and with structure.",
    "ISHER O8 Polyglot": "You are Isher O8 Polyglot, a culturally sensitive translation expert. Translate accurately between major world languages and explain nuances clearly. Maintain elegance and precision in tone.",
    "ISHER O9 Ledger": "You are Isher O9 Ledger, a strategic expert in finance, business operations, and management consulting. Answer clearly, insightfully, and with practical examples from real-world finance."
}

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    global uploaded_context
    file = request.files["file"]
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    if file.filename.endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = "\n".join([page.get_text() for page in doc])
    elif file.filename.endswith(".docx"):
        d = docx.Document(file)
        text = "\n".join([para.text for para in d.paragraphs])
    else:
        return jsonify({"error": "Unsupported file type"}), 400

    uploaded_context = text[:10000]  # Limit to 10k characters
    return jsonify({"extracted_text": uploaded_context})

@app.route("/chat", methods=["POST"])
def chat():
    global uploaded_context
    data = request.get_json()
    history = data.get("history", [])
    selected_model = data.get("selected_model", "Bolt O4 Nexus")

    system_prompt = SYSTEM_PROMPTS.get(selected_model, SYSTEM_PROMPTS["Isher 04 Nexus"])

    # Add document context
    if uploaded_context:
        system_prompt += f"\n\nYou also have access to the following information from a document:\n{uploaded_context}"

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
