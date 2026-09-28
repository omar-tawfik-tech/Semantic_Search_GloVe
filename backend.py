from pathlib import Path
import re
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(
    title="Semantic Search with GloVe API",
    description="Semantic search using 100-dimensional GloVe embeddings and cosine similarity.",
    version="1.0.0",
)

BASE_DIR = Path(__file__).resolve().parent
GLOVE_PATH = BASE_DIR / "models" / "glove.6B.100d.txt"

documents = [
    "Machine learning is a field of artificial intelligence.",
    "Deep learning uses neural networks to learn from data.",
    "Python is a popular programming language.",
    "Natural language processing helps computers understand human language.",
    "Computer vision allows machines to understand images.",
    "Neural networks are widely used in deep learning.",
    "Artificial intelligence is used in many real world applications.",
    "Natural language processing is used for text classification and translation.",
    "Machine learning algorithms can be used for prediction.",
    "Deep learning models require large amounts of training data.",
    "Omar Tawfik Is A Famous Programmer",
    "omar Tawfik Is A python instructor ",
    "graphic desiging is a filed talk about editng phots and videos",
    "graphic desiging is a filed learned in applied arts college",
    "mostafa hefny is the greatest graphic desginer",
    "ramez tamer is enginner and biggest ass in the country",
    "Electrical enginnering fucks ramez tamer",
    "ramez tamer has a small dick",
]

def text_clean(text: str):
    # The notebook intended lowercase + punctuation removal.
    # The original notebook had text.lower() without assignment and an unused variable.
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.split()

def load_glove(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"GloVe file not found: {path}. Put glove.6B.100d.txt inside models/."
        )

    embeddings = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            if len(values) != 101:
                continue
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings[word] = vector
    return embeddings

try:
    embeddings = load_glove(GLOVE_PATH)
    glove_error = None
except Exception as e:
    embeddings = {}
    glove_error = str(e)

def document_vector(tokens, embeddings):
    vectors = [embeddings[token] for token in tokens if token in embeddings]
    if not vectors:
        return np.zeros(100, dtype="float32")
    return np.mean(vectors, axis=0)

# Pre-compute document vectors once at startup instead of recalculating them per query.
document_vectors = [
    document_vector(text_clean(document), embeddings)
    for document in documents
]

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Text query")
    k: int = Field(default=3, ge=1, le=20, description="Number of results")

@app.get("/")
def root():
    return {"message": "Semantic Search with GloVe API is running"}

@app.get("/health")
def health():
    return {
        "status": "ok" if embeddings else "error",
        "glove_loaded": bool(embeddings),
        "vocabulary_size": len(embeddings),
        "documents": len(documents),
        "error": glove_error,
    }

@app.post("/search")
def search(request: SearchRequest):
    if not embeddings:
        raise HTTPException(status_code=500, detail=glove_error)

    query_tokens = text_clean(request.query)
    query_vector = document_vector(query_tokens, embeddings)

    if not np.any(query_vector):
        return {
            "query": request.query,
            "results": [],
            "message": "None of the query words were found in GloVe vocabulary."
        }

    scores = []
    for i, doc_vector in enumerate(document_vectors):
        if not np.any(doc_vector):
            score = 0.0
        else:
            score = float(
                cosine_similarity(
                    query_vector.reshape(1, -1),
                    doc_vector.reshape(1, -1)
                )[0][0]
            )
        scores.append((i, score))

    scores.sort(key=lambda x: x[1], reverse=True)

    results = [
        {
            "document": documents[index],
            "score": round(score, 4),
            "rank": rank,
        }
        for rank, (index, score) in enumerate(scores[:request.k], start=1)
    ]

    return {"query": request.query, "results": results}
