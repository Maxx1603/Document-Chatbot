from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from pinecone import Pinecone
from google import generativeai as genai
from groq import Groq
from dotenv import load_dotenv

import os
import uuid

# ---------------- LOAD ENV ----------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GOOGLE_API_KEY or not PINECONE_API_KEY or not GROQ_API_KEY:
    raise ValueError("Missing API keys in environment variables")

# ---------------- CONFIGURE CLIENTS ----------------
genai.configure(api_key=GOOGLE_API_KEY)

groq_client = Groq(api_key=GROQ_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("document-chatbot")

# ---------------- FASTAPI APP ----------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- REQUEST MODEL ----------------
class ChatRequest(BaseModel):
    question: str

# ---------------- EMBEDDING ----------------
def get_embedding(text: str):
    try:
        result = genai.embed_content(
            model="text-embedding-004",
            content=text
        )
        return result["embedding"]

    except Exception as e:
        print("EMBEDDING ERROR:", e)
        return None

# ---------------- SPLIT TEXT ----------------
def split_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks

# ---------------- HOME ----------------
@app.get("/")
def home():
    return {"message": "Backend Running"}

# ---------------- UPLOAD PDF ----------------
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    try:
        os.makedirs("uploads", exist_ok=True)

        file_path = f"uploads/{file.filename}"

        with open(file_path, "wb") as f:
            f.write(await file.read())

        reader = PdfReader(file_path)

        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        if not text.strip():
            return {"error": "No readable text found in PDF"}

        chunks = split_text(text)

        vectors = []

        for chunk in chunks:
            embedding = get_embedding(chunk)

            if embedding:
                vectors.append(
                    (
                        str(uuid.uuid4()),
                        embedding,
                        {"text": chunk}
                    )
                )

        if vectors:
            index.upsert(vectors=vectors)

        return {
            "message": "PDF uploaded successfully",
            "chunks_uploaded": len(vectors)
        }

    except Exception as e:
        print("UPLOAD ERROR:", e)
        return {"error": str(e)}

# ---------------- CHAT ----------------
@app.post("/chat")
def chat(data: ChatRequest):

    try:
        query_embedding = get_embedding(data.question)

        if not query_embedding:
            return {"error": "Embedding failed"}

        results = index.query(
            vector=query_embedding,
            top_k=10,
            include_metadata=True
        )

        matches = results.get("matches", [])

        if not matches:
            return {"answer": "No relevant data found."}

        context = "\n\n".join([
            m["metadata"]["text"] for m in matches
        ])

        prompt = f"""
You are an AI assistant.

Use ONLY the context below.

CONTEXT:
{context}

QUESTION:
{data.question}

ANSWER:
"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        answer = response.choices[0].message.content

        return {"answer": answer}

    except Exception as e:
        print("CHAT ERROR:", e)
        return {"error": str(e)}