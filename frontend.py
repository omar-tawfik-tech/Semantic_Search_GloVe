import streamlit as st
import requests

st.set_page_config(
    page_title="Semantic Search with GloVe",
    page_icon="🔎",
    layout="centered",
)

API_URL = "http://127.0.0.1:8000"

st.title("🔎 Semantic Search with GloVe")
st.write("Search the corpus using 100-dimensional GloVe embeddings and cosine similarity.")

query = st.text_input(
    "Enter your query",
    placeholder="e.g. machine learning algorithms"
)

k = st.slider("Number of results", min_value=1, max_value=10, value=3)

if st.button("Search", type="primary"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        try:
            response = requests.post(
                f"{API_URL}/search",
                json={"query": query, "k": k},
                timeout=60,
            )

            if response.status_code != 200:
                st.error(response.json().get("detail", response.text))
            else:
                data = response.json()
                results = data.get("results", [])

                if not results:
                    st.info(data.get("message", "No results found."))
                else:
                    st.subheader("Results")
                    for result in results:
                        st.markdown(f"### #{result['rank']} — Score: {result['score']:.4f}")
                        st.write(result["document"])
                        st.divider()

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to FastAPI. Start the backend first:\n\n"
                "`uvicorn backend:app --reload`"
            )
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")

st.caption("Backend: FastAPI • Frontend: Streamlit • Embeddings: GloVe 6B 100d")
