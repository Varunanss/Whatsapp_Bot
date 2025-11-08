# backend/app.py
import os, base64, io, traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai

# Load .env
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

app = Flask(__name__)
CORS(app)

def ensure_model():
    if not GEMINI_API_KEY:
        raise RuntimeError("❌ GEMINI_API_KEY is missing. Add it to backend/.env")
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(GEMINI_MODEL)

model = ensure_model()

@app.get("/health")
def health():
    return {"ok": True, "model": GEMINI_MODEL}

def decode_image(data_url):
    if not data_url or not data_url.startswith("data:"):
        return None
    _, b64 = data_url.split(",", 1)
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")

@app.post("/api/chat")
def chat():
    try:
        data = request.get_json() or {}
        message = (data.get("message") or "").strip()
        history = data.get("history") or []
        image_data = data.get("image")

        if not message and not image_data:
            return jsonify({"reply": "Please send a message."}), 400

        system = (
            "You are Saathi, an agriculture support assistant. "
            "Answer briefly and practically. Provide Telugu if helpful."
        )

        parts = [system]

        for h in history:
            parts.append(h["content"])

        if message:
            parts.append(message)

        img = decode_image(image_data)
        if img:
            parts.append(img)

        response = model.generate_content(parts)
        reply = response.text.strip() if response and response.text else "Could not generate a response."
        return jsonify({"reply": reply})

    except Exception as e:
        print("🔥 ERROR:", e)
        traceback.print_exc()
        return jsonify({"reply": "Backend error. Try again."}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)
