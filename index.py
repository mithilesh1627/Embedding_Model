from flask import Flask, request, jsonify
import tensorflow_hub as hub
from flask_cors import CORS
import numpy
app = Flask(__name__)
CORS(app)


embed_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

@app.route('/embed', methods=['POST'])
def get_embedding():
    try:
        data = request.json
        text = data.get("text")

        if not text:
            return jsonify({"error": "Text input is required"}), 400

        embedding = embed_model([text])[0].numpy().tolist()

        return jsonify({"embedding": embedding})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
