import os
import faiss
import numpy as np
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_pymongo import PyMongo
from sentence_transformers import SentenceTransformer
from bson import ObjectId

app = Flask(__name__)
CORS(app)

# ✅ Load OpenRouter API Key from Environment Variable
API_KEY = "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" # Make sure to set this in your environment
if not API_KEY:
    raise ValueError("❌ API key not found in environment variables!")

# ✅ OpenRouter API Endpoint
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# ✅ MongoDB Configuration
app.config["MONGO_URI"] = "mongodb+srv://Jarvis:am1qJlDpmSt7jkc5@ml-cluster.4ltlg.mongodb.net/vector_DB"
mongo = PyMongo(app)
collection = mongo.db.embeddings

# ✅ Load Sentence Transformer Model
model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Initialize FAISS Index (384-dim embeddings)
dimension = 384
index = faiss.IndexFlatL2(dimension)

# ✅ Load existing embeddings from MongoDB into FAISS
def load_existing_embeddings():
    print("📢 Loading embeddings from MongoDB into FAISS...")
    all_docs = list(collection.find({}, {"embedding": 1}))

    if all_docs:
        embeddings = [doc["embedding"] for doc in all_docs]
        embeddings = np.array(embeddings, dtype=np.float32)
        index.add(embeddings)
        print(f"✅ Loaded {len(all_docs)} embeddings into FAISS.")

load_existing_embeddings()

@app.route('/add', methods=['POST'])
def add_embedding():
    try:
        data = request.get_json()
        title = data.get("title")
        doc_type = data.get("type")
        link = data.get("link")
        user_id = data.get("userID")

        if not all([title, doc_type, link, user_id]):
            return jsonify({"error": "All fields (title, type, link, userID) are required"}), 400

        try:
            user_id = ObjectId(user_id)
        except:
            return jsonify({"error": "Invalid userID format"}), 400

        embedding = model.encode(title).astype(np.float32).tolist()

        doc = {"title": title, "type": doc_type, "link": link, "userID": user_id, "embedding": embedding}
        result = collection.insert_one(doc)
        doc_id = str(result.inserted_id)

        index.add(np.array([embedding], dtype=np.float32))

        print(f"✅ Added document: {doc}")

        return jsonify({"message": "Embedding added successfully", "id": doc_id})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/search', methods=['POST'])
def search_embedding():
    try:
        data = request.get_json()
        query_text = data.get("query")

        if not query_text:
            return jsonify({"error": "Query text is required"}), 400

        query_embedding = model.encode(query_text).reshape(1, -1).astype(np.float32)

        D, I = index.search(query_embedding, k=1)

        results = []
        for idx in I[0]:
            if idx == -1:
                continue

            doc = collection.find_one({}, {"title": 1, "type": 1, "link": 1, "userID": 1, "_id": 0}, skip=int(idx))

            if doc:
                results.append({
                    "title": doc["title"],
                    "type": doc["type"],
                    "link": doc["link"],
                    "userID": str(doc["userID"])
                })

        # ✅ Use OpenRouter API
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        openrouter_payload = {
            "model": "mistralai/mistral-7b-instruct:free",  # Using Gemma
            "messages": [
                {"role": "system", "content": "You are an AI assistant that helps users find the best search results."},
                {"role": "user",
                 "content": f"Given the search query: '{query_text}', and these results: {results}, suggest the best match in a short, clear sentence. Avoid bullet points, excessive formatting, and unnecessary explanations."}
            ]
        }

        response = requests.post(OPENROUTER_URL, json=openrouter_payload, headers=headers)

        if response.status_code != 200:
            return jsonify({"error": "Failed to get AI response", "details": response.json()}), 500

        # ✅ Clean the response output
        ai_thoughts = response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip().replace(
            "\n\n", "\n")

        return jsonify({"results": results, "AI_Thoughts": ai_thoughts})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
