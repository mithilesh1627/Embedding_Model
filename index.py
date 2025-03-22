from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app)
model = SentenceTransformer("all-MiniLM-L6-v2")

@app.route('/', methods=['POST'])
def get_embedding():
    try:
        if request.content_type != "application/json":
            return jsonify({"error": "Invalid Content-Type. Expected application/json"}), 415

        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({"error": "Text input is required"}), 400

        text = data["text"]
        embedding = model.encode(text).tolist()  # Generate actual embeddings

        return jsonify({"embedding": embedding})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
