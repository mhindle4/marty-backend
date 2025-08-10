# app.py
import os
import time
import json
import random
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# --- Text model: Google Generative AI (Gemini) ---
import google.generativeai as genai

# --- Voice: ElevenLabs (simple REST call, no SDK needed) ---
import requests

# -------------------------
# Config & setup
# -------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
CLONED_VOICE_ID = os.getenv("CLONED_VOICE_ID")  # e.g. "AbcDef123..."
ELEVEN_MODEL_ID = os.getenv("ELEVEN_MODEL_ID", "eleven_multilingual_v2")  # safe default

if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY env var")
if not ELEVENLABS_API_KEY:
    raise RuntimeError("Missing ELEVENLABS_API_KEY env var")
if not CLONED_VOICE_ID:
    raise RuntimeError("Missing CLONED_VOICE_ID env var")

genai.configure(api_key=GEMINI_API_KEY)

# Flask app + CORS
app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

# Ensure the audio output directory exists
AUDIO_DIR = Path("static") / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Personality controls (edit these to tune Marty)
# -------------------------
PERSONA = """
You are Marty — a friendly, upbeat AI with a clear, concise way of explaining things.
Your tone is warm, encouraging, and practical. Use plain British English.
Keep replies focused: usually 2–4 short sentences unless the user asks for detail.
Avoid jargon unless asked; avoid over-apologising or sounding robotic.
If something is unclear, briefly ask a helpful clarifying question.
If you don’t know, say so briefly and suggest a next step.
You enjoy rock music.
you think locically.
you tell lots of jokes.
you can be sarcastic.
you sometimes cheekily embelish the truth.
"""

FEWSHOTS = """
User: Hi, who are you?
Marty: Right then, I’m Marty—here to help. What are we tackling today?

User: Can you explain this simply?
Marty: Good shout. Here’s the plain-English version: …

User: Thanks!
Marty: No worries—happy to help.
"""

# Light seasoning of catchphrases (applied after model reply)
CATCHPHRASES = ["Right then,", "Good shout.", "Let’s sort that.", "No worries.","No problem.","Got it!."]

# Model behaviour (tweak to taste)
TEMPERATURE = float(os.getenv("MARTY_TEMP", "0.7"))         # 0.2–0.4 = precise; 0.7–1.0 = lively
MAX_OUTPUT_TOKENS = int(os.getenv("MARTY_MAX_TOKENS", "300"))

# Add a catchphrase to the start of some replies (set to 0 to disable)
CATCHPHRASE_PROB = float(os.getenv("MARTY_PHRASE_PROB", "0.30"))

# -------------------------
# Routes (serve pages if present)
# -------------------------
@app.route("/")
def home():
    index_path = Path("index.html")
    if index_path.exists():
        return send_from_directory(".", "index.html")
    return "Marty backend is running. Try /chat.html"

@app.route("/chat.html")
def chat_page():
    return send_from_directory(".", "chat.html")

@app.route("/about.html")
def about_page():
    return send_from_directory(".", "about.html")

@app.route("/contact.html")
def contact_page():
    return send_from_directory(".", "contact.html")

# -------------------------
# Helpers
# -------------------------
def build_prompt(user_msg: str, history=None) -> str:
    """
    You can extend this later to include a running conversation history.
    For now we keep it simple and robust.
    """
    return f"""{PERSONA}

{FEWSHOTS}

Conversation:
User: {user_msg}
Marty:"""

def lightly_season(text: str) -> str:
    """Optionally prepend a catchphrase (not every time)."""
    if not text:
        return text
    try:
        if CATCHPHRASES and random.random() < CATCHPHRASE_PROB:
            # avoid doubling if the model already started with one
            if not any(text.startswith(p) for p in CATCHPHRASES):
                return f"{random.choice(CATCHPHRASES)} {text}"
    except Exception:
        pass
    return text

def tts_to_file(text: str) -> str | None:
    """Send text to ElevenLabs, stream MP3 to disk, return /static url or None."""
    try:
        eleven_url = f"https://api.elevenlabs.io/v1/text-to-speech/{CLONED_VOICE_ID}/stream"
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": ELEVEN_MODEL_ID,
            # You can tune the voice a bit here if you want:
            # "voice_settings": {"stability": 0.5, "similarity_boost": 0.8}
        }

        with requests.post(eleven_url, headers=headers, data=json.dumps(payload), stream=True, timeout=60) as r:
            r.raise_for_status()
            filename = f"marty_{int(time.time()*1000)}.mp3"
            out_path = AUDIO_DIR / filename
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return f"/static/audio/{filename}"
    except Exception as e:
        print("ElevenLabs TTS error:", repr(e))
        return None

# -------------------------
# Core chat endpoint
# -------------------------
@app.route("/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # 1) Text reply from Gemini
    try:
        model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
        prompt = build_prompt(user_message)
        resp = model.generate_content(
            prompt,
            generation_config={
                "temperature": TEMPERATURE,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
            },
        )
        bot_text = (resp.text or "").strip()
        if not bot_text:
            bot_text = "I’m here—how can I help?"
    except Exception as e:
        print("Gemini error:", repr(e))
        bot_text = "Sorry, I hit a snag generating that. Want to try a different way?"

    # 2) Optional seasoning with catchphrases
    bot_text = lightly_season(bot_text)

    # 3) TTS via ElevenLabs (best-effort)
    audio_url = tts_to_file(bot_text)

    return jsonify({"text": bot_text, "audioUrl": audio_url})

# -------------------------
# Run (dev) / Render (prod)
# -------------------------
if __name__ == "__main__":
    # Use PORT env var if present (Render provides this), else default locally
    port = int(os.getenv("PORT", "5000"))
    app.run(debug=True, host="0.0.0.0", port=port)
