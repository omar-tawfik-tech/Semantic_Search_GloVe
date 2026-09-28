# Semantic Search with GloVe

A small semantic-search project based on the same model from `Semantic_Search_with_GloVe.ipynb`, exposed through:

- **FastAPI** → backend/API
- **Streamlit** → frontend/UI
- **GloVe 6B 100d** → word embeddings
- **Cosine similarity** → ranking

## Project structure

```text
Semantic_Search_GloVe/
├── Model.ipynb
├── backend.py
├── frontend.py
├── requirements.txt
├── README.md
├── .gitignore
└── models/
    └── glove.6B.100d.txt   # add this file yourself
```

## 1. Install

```bash
python -m venv First_Rag
First_Rag\Scripts\activate
pip install -r requirements.txt
```

## 2. Add GloVe

Download **GloVe 6B** and extract:

```text
glove.6B.100d.txt
```

Put it here:

```text
models/glove.6B.100d.txt
```

The full GloVe file is intentionally not included in Git because it is large.

## 3. Start FastAPI

From the project folder:

```bash
uvicorn backend:app --reload
```

API:

```text
http://127.0.0.1:8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

## 4. Start Streamlit

Open another terminal:

```bash
streamlit run frontend.py
```

Then open the Streamlit URL shown in the terminal.

## Important note about the original notebook

The original `text_clean` function contains:

```python
text.lower()
rext = re.sub(...)
tokens = text.split()
```

`text.lower()` does not modify the string in-place, and the cleaned `rext` variable was never used.

For deployment, `backend.py` intentionally fixes this to:

```python
text = text.lower()
text = re.sub(r"[^a-zA-Z\s]", "", text)
return text.split()
```

The actual semantic-search approach remains the same:

1. Clean query.
2. Convert query words to GloVe vectors.
3. Average the word vectors.
4. Average each document's word vectors.
5. Calculate cosine similarity.
6. Sort documents by similarity.
7. Return top-k results.
