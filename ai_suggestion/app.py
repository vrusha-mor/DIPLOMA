from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import logging
import re
import random
import time

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

GEMINI_API_KEY = "AIzaSyCbtnrQQ5xWNSMfMxiY_9iDABEZn-y6HTc"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

@app.route('/', methods=['POST'])
def root():
    return jsonify({"error": "Use /get_suggestion endpoint for suggestions"}), 400

@app.route('/get_suggestion', methods=['POST'])
def get_suggestion():
    try:
        data = request.json
        app.logger.debug(f"Received request: {data}")
        energy_level = data.get("energyLevel")
        preferences = data.get("preferences", [])

        if not energy_level:
            return jsonify({"error": "Energy level is required"}), 400

        random_token = random.randint(10000, 99999)
        timestamp = int(time.time())

        prompt = f"""You are a mental wellness assistant. 
        My energy level is {energy_level}. 
        I prefer {', '.join(preferences) if preferences else 'any'} activities. 
        Suggest 3 different activities with creative, uplifting descriptions. 
        Each activity should be exactly one paragraph and strictly under 50 words. 
        Do not use bullet points or symbols. Use plain language only. Token:{random_token}-{timestamp}"""

        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 1.0,
                "topK": 50,
                "topP": 0.95
            }
        }

        response = requests.post(GEMINI_API_URL, json=payload)
        app.logger.debug(f"Gemini API response: {response.status_code}, {response.text}")

        if response.status_code == 503:
            app.logger.warning("Gemini API is overloaded.")
            return jsonify({"error": "Gemini API is overloaded. Please try again later."}), 503

        if not response.ok:
            return jsonify({
                "error": f"API Error: {response.status_code} - {response.text}"
            }), 500

        try:
            response_data = response.json()
            raw_text = response_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()

            # Sanitize output and split suggestions
            raw_text = re.sub(r"[*•\-►●▪]", "", raw_text)
            suggestions = re.split(r"\n\s*\n|\d\.\s*", raw_text)
            suggestions = [s.strip() for s in suggestions if s.strip()]

            # Limit each suggestion to ~50 words
            limited_suggestions = []
            for s in suggestions[:3]:  # Max 3 suggestions
                words = s.split()
                trimmed = " ".join(words[:50])
                limited_suggestions.append(trimmed)

            final_output = "\n\n".join(f"{i+1}. {s}" for i, s in enumerate(limited_suggestions))
            app.logger.debug(f"Final Limited Suggestion: {final_output}")

            return jsonify({"suggestion": final_output})

        except ValueError as ve:
            app.logger.error(f"JSON decode error: {ve}")
            return jsonify({"error": "Invalid response format from Gemini API."}), 500

    except Exception as e:
        app.logger.exception("Unexpected error occurred")
        return jsonify({"error": "Internal server error. Please try again later."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
