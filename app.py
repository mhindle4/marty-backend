# app.py
import os
import time
import json
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

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

# Ensure the audio output directory exists
AUDIO_DIR = Path("static") / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Persona / system prompt for Marty (tweak freely)
PERSONA = (
    "You are Marty, an approachable, upbeat AI assistant with a friendly, "
    "concise style. Keep answers helpful and clear for beginners. "
    "Use plain English and avoid jargon unless asked."
)

# -------------------------
# Routes (serve pages if you want)
# -------------------------
@app.route("/")
def home():
    # Optional: if you have an index.html in the same folder
    index_path = Path("index.html")
    if index_path.exists():
        return send_from_directory(".", "index.html")
    return "Marty backend is running. Go to /chat.html"

@app.route("/chat.html")
def chat_page():
    # Serve the chat UI file you pasted earlier
    return send_from_directory(".", "chat.html")

@app.route("/about.html")
def about_page():
    return send_from_directory(".", "about.html")

@app.route("/contact.html")
def contact_page():
    return send_from_directory(".", "contact.html")



# -------------------------
# Core chat endpoint
# -------------------------
@app.route("/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # 1) Get text reply from Gemini
    try:
        model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
        prompt = f"{PERSONA}\n\nUser: {user_message}\nMarty:"
        resp = model.generate_content(prompt)
        bot_text = (resp.text or "").strip()
        if not bot_text:
            bot_text = "I’m here! How can I help?"
    except Exception as e:
        # Fallback so the user gets *something*
        bot_text = "Sorry, I had trouble generating a response."
        print("Gemini error:", repr(e))

    # 2) Convert text -> speech via ElevenLabs (stream to file)
    audio_url = None
    try:
        eleven_url = f"https://api.elevenlabs.io/v1/text-to-speech/{CLONED_VOICE_ID}/stream"
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
        }
        payload = {
            "text": bot_text,
            "model_id": ELEVEN_MODEL_ID,
            # You can add voice settings if you like:
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
        audio_url = f"/static/audio/{filename}"
    except Exception as e:
        print("ElevenLabs TTS error:", repr(e))
        # audio_url remains None; the UI will still show text

    return jsonify({"text": bot_text, "audioUrl": audio_url})


# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    # host=0.0.0.0 allows access from other devices on your network
    app.run(debug=True, host="0.0.0.0", port=5000)

