from flask import Flask, request, render_template, Response
from openai import OpenAI
from dotenv import load_dotenv
import os
import re

load_dotenv()
app = Flask(__name__)

# ✅ Groq-compatible OpenAI setup
groq = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# ✅ System prompt per model
SYSTEM_PROMPTS = {
    "Bolt O4 Nexus": "You are Bolt O4 Nexus, an efficient, friendly AI assistant that helps with any task in a concise and smart way. Always be helpful and clear.",
    "Bolt O5 Forge": "You are Bolt O5 Forge, a highly skilled coding and developer assistant. You answer technically, clearly, and with precision.",
    "Bolt O6 Vita": "You are Bolt O6 Vita, a compassionate and knowledgeable health and wellness assistant. Offer personalised guidance on nutrition, fitness, mental health, and lifestyle choices in a warm, clear tone.",
    "Bolt O7 Quill": "You are Bolt O7 Quill, a professional writing and academic assistant. You write formally, elegantly, and with structure.",
    "Bolt O8 Polyglot": "You are Bolt O8 Polyglot, a culturally sensitive translation expert. Translate accurately between major world languages and explain nuances clearly. Maintain elegance and precision in tone."
}


def clean_token(token):
    return re.sub(r"[*_`•▶️➡️➤]+", "", token)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    history = data.get("history", [])
    model_choice = data.get("selected_model", "Bolt O4 Nexus")

    def stream():
        try:
            messages = [{"role": "system", "content": SYSTEM_PROMPTS.get(model_choice, SYSTEM_PROMPTS["Bolt O4 Nexus"])}]
            messages += [{"role": m["role"], "content": m["content"]} for m in history]

            response = groq.chat.completions.create(
                model="llama3-70b-8192",
                messages=messages,
                temperature=0.7,
                stream=True
            )

            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    yield clean_token(token)

        except Exception as e:
            yield "\n⚠️ Error streaming response: " + str(e)

    return Response(stream(), mimetype='text/plain')

if __name__ == '__main__':
    app.run(debug=True)
