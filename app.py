from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.genai as genai
from dotenv import load_dotenv
import os
import csv

load_dotenv()

app = Flask(__name__)

# Configure Gemini
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Load data
docs = []
with open("/workspaces/dev/qa_data (1).csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        docs.append(row['answer'])

# Embeddings
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embed_model.encode(docs).astype("float32")

index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    question = request.json["question"]

    q_embedding = embed_model.encode([question]).astype("float32")
    _, idx = index.search(q_embedding, 2)

    context = " ".join([docs[i] for i in idx[0]])
    answer = context

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
