from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from pinecone import Pinecone
from google import genai
from groq import Groq
from dotenv import load_dotenv

import os
import uuid

# ---------------- LOAD ENV ----------------
load_dotenv()

# ---------------- API KEYS ----------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---------------- CHECK KEYS ----------------
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

# ---------------- APP ----------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- CLIENTS ----------------
client = genai.Client(
    api_key=GOOGLE_API_KEY
)

groq_client = Groq(
    api_key=GROQ_API_KEY
)

pc = Pinecone(
    api_key=PINECONE_API_KEY
)

index = pc.Index("document-chatbot")

# ---------------- REQUEST MODEL ----------------
class ChatRequest(BaseModel):
    question: str

# ---------------- EMBEDDING ----------------
def get_embedding(text: str):

    try:

        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text
        )

        return result.embeddings[0].values

    except Exception as e:

        print("EMBEDDING ERROR:", e)

        return None

# ---------------- TEXT SPLITTING ----------------
def split_text(text, chunk_size=800, overlap=150):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

# ---------------- HOME ----------------
@app.get("/")
def home():

    return {
        "message": "Backend Running"
    }

# ---------------- PDF UPLOAD ----------------
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    try:

        # create uploads folder
        os.makedirs("uploads", exist_ok=True)

        file_path = f"uploads/{file.filename}"

        # save uploaded pdf
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # read pdf
        reader = PdfReader(file_path)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        # check empty text
        if text.strip() == "":

            return {
                "error": "No readable text found in PDF"
            }

        # split text
        chunks = split_text(text)

        vectors = []

        # create embeddings
        for chunk in chunks:

            embedding = get_embedding(chunk)

            if embedding:

                vectors.append(
                    (
                        str(uuid.uuid4()),
                        embedding,
                        {
                            "text": chunk
                        }
                    )
                )

        # upload vectors
        if vectors:

            index.upsert(vectors=vectors)

        return {
            "message": "PDF uploaded successfully",
            "chunks_uploaded": len(vectors)
        }

    except Exception as e:

        print("UPLOAD ERROR:", e)

        return {
            "error": str(e)
        }

# ---------------- CHAT ----------------
@app.post("/chat")
def chat(data: ChatRequest):

    try:

        # question embedding
        query_embedding = get_embedding(data.question)

        if not query_embedding:

            return {
                "error": "Embedding failed"
            }

        # search pinecone
        results = index.query(
            vector=query_embedding,
            top_k=10,
            include_metadata=True
        )

        matches = results.get("matches", [])

        if not matches:

            return {
                "answer": "No relevant data found."
            }

        # summary detection
        summary_keywords = [
            "summary",
            "summarize",
            "explain",
            "overview",
            "describe",
            "about document"
        ]

        question_lower = data.question.lower()

        is_summary_question = any(
            keyword in question_lower
            for keyword in summary_keywords
        )

        # filter matches
        if is_summary_question:

            filtered_matches = matches

        else:

            filtered_matches = []

            for match in matches:

                score = match.get("score", 0)

                if score > 0.30:
                    filtered_matches.append(match)

        if len(filtered_matches) == 0:

            return {
                "answer": "Answer not found in document."
            }

        # build context
        context = "\n\n".join([
            match["metadata"]["text"]
            for match in filtered_matches
        ])

        # prompt
        prompt = f"""
You are an AI document assistant.

RULES:
- Answer ONLY from the document context
- Give direct answers
- If names exist, return names exactly
- Do NOT invent information
- If user asks for summary, summarize the document
- If answer does not exist, say:
  "Answer not found in document."

DOCUMENT CONTEXT:
{context}

QUESTION:
{data.question}

ANSWER:
"""

        # ask Groq
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer
        }

    except Exception as e:

        print("CHAT ERROR:", e)

        return {
            "error": str(e)
        }