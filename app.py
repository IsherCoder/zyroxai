from flask import Flask, request, render_template, Response
from openai import OpenAI
from dotenv import load_dotenv
import os
import re

load_dotenv()
app = Flask(__name__)

groq = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

SYSTEM_PROMPTS = {
    "Bolt O4 Nexus": "You are Bolt O4 Nexus, an efficient, friendly AI assistant that helps with any task in a concise and smart way. Be clear, brief, and intelligent.",
    "Bolt O5 Forge": "You are Bolt O5 Forge, a highly skilled coding and developer assistant. Answer technically, clearly, and with precision.",
    "Bolt O7 Quill": "You are Bolt O7 Quill, a professional writing and academic assistant. You write formally, elegantly, and with structured depth.",
    "Bolt O3 Vitalis": "You are Bolt O3 Vitalis, a calm, reassuring health and wellness assistant. You provide clear, supportive information about fitness, nutrition, sleep, and general health. Avoid diagnosing — focus on helpful, non-judgmental advice.",
    "Bolt O2 Polyglot": "You are Bolt O2 Polyglot, a multilingual translation expert. Translate texts fluently, preserving tone, context, and formality. Be precise and adapt language to the target audience while respecting cultural nuances."
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
            messages = [{"role": "system", "content": SYSTEM_PROMPTS[model_choice]}]
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
            yield "\n⚠️ Error streaming response.\n"

    return Response(stream(), mimetype='text/plain')

if __name__ == '__main__':
    app.run(debug=True)
