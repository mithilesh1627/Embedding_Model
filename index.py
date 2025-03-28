import os
import faiss
import numpy as np
import requests
import markdown
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_pymongo import PyMongo
from sentence_transformers import SentenceTransformer
from youtube_transcript_api import YouTubeTranscriptApi
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
from bson.objectid import ObjectId  # ✅ Import ObjectId for MongoDB

app = Flask(__name__)
CORS(app)

# ✅ Load OpenRouter API Key from Environment Variable
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
print(OPENROUTER_API_KEY)
if not OPENROUTER_API_KEY:
    raise ValueError("❌ OpenRouter API key not found in environment variables!")

# ✅ MongoDB Configuration
app.config["MONGO_URI"] = "mongodb+srv://Jarvis:am1qJlDpmSt7jkc5@ml-cluster.4ltlg.mongodb.net/vector_DB"
mongo = PyMongo(app)
collection = mongo.db.embeddings

# ✅ Load Sentence Transformer Model
model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Initialize FAISS Index (384-dim embeddings)
dimension = 384
index = faiss.IndexFlatL2(dimension)


def load_existing_embeddings():
    """Loads embeddings from MongoDB into FAISS index."""
    print("📢 Loading embeddings from MongoDB into FAISS...")
    all_docs = list(collection.find({}, {"embedding": 1}))

    if all_docs:
        embeddings = [doc["embedding"] for doc in all_docs]
        embeddings = np.array(embeddings, dtype=np.float32)
        index.add(embeddings)
        print(f"✅ Loaded {len(all_docs)} embeddings into FAISS.")


load_existing_embeddings()


def clean_summary(summary):
    """Removes newline characters and extra spaces from the summary."""
    return summary.replace("\n", " ").replace("  ", " ")


def extract_text_from_webpage(url):
    """Extracts text content from a webpage."""
    try:
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = [p.get_text() for p in soup.find_all('p')]
        return ' '.join(paragraphs)
    except Exception as e:
        return f"Failed to extract content: {str(e)}"


def extract_youtube_transcript(url):
    """Extracts transcript from a YouTube video."""
    try:
        parsed_url = urlparse(url)
        video_id = parse_qs(parsed_url.query).get('v', [None])[0]
        if not video_id:
            return "Invalid YouTube URL."

        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = ' '.join([entry['text'] for entry in transcript])
        return transcript_text
    except Exception as e:
        return f"Failed to extract transcript: {str(e)}"


def summarize_with_mistral(content):
    """Summarizes content using Mistral AI."""
    try:
        if not OPENROUTER_API_KEY:
            return "❌ OpenRouter API key is missing."

        prompt = f"Summarize this content: {content}"

        payload = {
            "model": "deepseek/deepseek-chat-v3-0324:free",
            "messages": [{"role": "system", "content": "You are an AI assistant that summarizes content concisely."},
                         {"role": "user", "content": prompt}]
        }
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}

        response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)
        response_json = response.json()
        summary_md = response_json.get("choices", [{}])[0].get("message", {}).get("content", "Summary not available.")

        # Convert Markdown to HTML and clean summary
        formatted_summary = markdown.markdown(summary_md)
        return clean_summary(formatted_summary)

    except Exception as e:
        return f"Failed to summarize: {str(e)}"


@app.route('/search', methods=['POST'])
def search_embedding():
    """Searches for similar embeddings and summarizes relevant content."""
    try:
        data = request.get_json()
        query_text = data.get("query")

        if not query_text:
            return jsonify({"error": "Query text is required"}), 400

        if "twitter.com" in query_text or "x.com" in query_text:
            content = extract_tweet_text(query_text)
            summary_html = summarize_with_mistral(content)
            return jsonify({"source": "Twitter", "summary": summary_html})

        query_embedding = model.encode(query_text).reshape(1, -1).astype(np.float32)
        D, I = index.search(query_embedding, k=1)

        results = []
        for idx in I[0]:
            if idx == -1:
                continue

            doc = collection.find_one({}, {"title": 1, "type": 1, "link": 1, "userID": 1, "_id": 0}, skip=int(idx))

            if doc:
                results.append(doc)

        if not results:
            return jsonify({"error": "No relevant results found."}), 404

        best_result = results[0]
        link = best_result.get("link", "")

        if "youtube.com" in link:
            content = extract_youtube_transcript(link)
        else:
            content = extract_text_from_webpage(link)

        summary_html = summarize_with_mistral(content)

        # ✅ Clean the summary before returning
        best_result["summary"] = clean_summary(summary_html)

        return jsonify({"results": results})  # ✅ Removed "best_result_summary"

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/add', methods=['POST'])
def add_embedding():
    """Adds a new document with its embedding to MongoDB and FAISS."""
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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
