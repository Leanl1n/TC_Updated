import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List
import pandas as pd

def load_model(model_name: str, device: str):
    from sentence_transformers import SentenceTransformer
    try:
        model = SentenceTransformer(model_name, device=device)
        return model
    except Exception as e:
        print(f"Error loading model '{model_name}': {str(e)}")
        return None

def preprocess_texts(texts):
    return [
        str(text) 
        for text in texts 
        if text and not pd.isna(text) and not str(text).isspace()
    ]

def create_embeddings(self, texts: List[str], batch_size: int) -> np.ndarray:
    text_map = {
        i: str(text) 
        for i, text in enumerate(texts) 
        if text and not pd.isna(text) and not str(text).isspace()
    }
    unique_texts = list(set(text_map.values()))
    
    embeddings = []
    with ThreadPoolExecutor(max_workers=min(4, len(unique_texts))) as executor:
        futures = []
        for i in range(0, len(unique_texts), batch_size):
            batch = unique_texts[i:i + batch_size]
            futures.append(
                executor.submit(self.model.encode, batch, show_progress_bar=False)
            )
        
        embeddings = []
        for future in futures:
            embeddings.extend(future.result())

    text_to_embedding = dict(zip(unique_texts, embeddings))
    embedding_dim = embeddings[0].shape[0] if embeddings else self.model.get_sentence_embedding_dimension()
    
    return np.array([
        text_to_embedding.get(text_map.get(i), np.zeros(embedding_dim))
        for i in range(len(texts))
    ])

def calculate_cosine_similarity(self, embeddings: np.ndarray, batch_size: int = 1000) -> np.ndarray:
    n = len(embeddings)
    cosine_sim = np.zeros((n, n), dtype=np.float32)
    
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        batch = embeddings[i:end]
        cosine_sim[i:end] = cosine_similarity(batch, embeddings)
    
    return cosine_sim