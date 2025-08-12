# intent_model.py
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small, fast

def load_sbert(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    # You can wrap this with st.cache_resource in the Streamlit layer.
    return SentenceTransformer(model_name)

def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 64, show_progress_bar: bool = False) -> np.ndarray:
    # Normalize so cosine similarity works nicely later
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress_bar,
    )
