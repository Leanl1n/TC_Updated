import os
import numpy as np
import pandas as pd
import torch
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import warnings

class TextAnalyzer:
    def __init__(self):
        # Suppress warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        
        # Determine device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model_names = [
            'paraphrase-MiniLM-L6-v2',  # Lighter model, try first
            'paraphrase-MPNet-base-v2',  # More powerful but heavier
            'distilbert-base-nli-mean-tokens'  # Fallback option
        ]
        
        self.model = None
        last_error = None
        
        for model_name in model_names:
            try:
                st.info(f"Attempting to load model: {model_name}")
                # Initialize model with device specification
                self.model = SentenceTransformer(model_name, device=self.device)
                st.success(f"Successfully loaded model: {model_name}")
                break
                
            except Exception as e:
                last_error = str(e)
                st.warning(f"Failed to load {model_name}: {str(e)}")
                continue
        
        if self.model is None:
            st.error(f"Failed to initialize any model. Last error: {last_error}")
            raise RuntimeError("Could not initialize any model")
        
        self._embedding_cache: Dict[str, np.ndarray] = {}

    @lru_cache(maxsize=1000)
    def _get_embedding(self, text: str) -> np.ndarray:
        return self.model.encode([text], show_progress_bar=False)[0]

    def create_embeddings(self, texts: List[str], batch_size: int, progress_callback=None) -> np.ndarray:
        text_map = {
            i: str(text) 
            for i, text in enumerate(texts) 
            if text and not pd.isna(text) and not str(text).isspace()
        }
        unique_texts = list(set(text_map.values()))
        total_batches = (len(unique_texts) + batch_size - 1) // batch_size
        
        embeddings = []
        with ThreadPoolExecutor(max_workers=min(4, len(unique_texts))) as executor:
            futures = []
            for i in range(0, len(unique_texts), batch_size):
                batch = unique_texts[i:i + batch_size]
                futures.append(
                    executor.submit(self.model.encode, batch, show_progress_bar=False)
                )
                if progress_callback:
                    progress = (i + len(batch)) / len(unique_texts)
                    progress_callback(progress)
            
            embeddings = []
            for future in futures:
                embeddings.extend(future.result())

        text_to_embedding = dict(zip(unique_texts, embeddings))
        embedding_dim = embeddings[0].shape[0] if embeddings else self.model.get_sentence_embedding_dimension()
        
        return np.array([
            text_to_embedding.get(text_map.get(i), np.zeros(embedding_dim))
            for i in range(len(texts))
        ])

    def calculate_cosine_similarity(self, embeddings: np.ndarray, batch_size: int = 1000, progress_callback=None) -> np.ndarray:
        n = len(embeddings)
        cosine_sim = np.zeros((n, n), dtype=np.float32)
        
        total_batches = (n + batch_size - 1) // batch_size
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            batch = embeddings[i:end]
            cosine_sim[i:end] = cosine_similarity(batch, embeddings)
            
            if progress_callback:
                progress = (i + len(batch)) / n
                progress_callback(progress)
        
        return cosine_sim

    def _process_threshold(
        self, 
        df: pd.DataFrame,
        text_column: str,
        threshold: float,
        cosine_sim: np.ndarray
    ) -> Tuple[Dict, pd.Series]:
        n_samples = len(df)
        topics = np.full(n_samples, -1)
        current_topic = 1
        empty_topic = 0
        
        empty_mask = df[text_column].isna() | df[text_column].str.isspace()
        topics[empty_mask] = empty_topic
        
        first_valid = np.where(topics == -1)[0][0]
        topics[first_valid] = current_topic
        
        for i in range(1, n_samples):
            if topics[i] != -1:
                continue
                
            similar_mask = (topics[:i] > 0) & (cosine_sim[i, :i] >= threshold)
            if similar_mask.any():
                topics[i] = topics[:i][similar_mask][0]
            else:
                current_topic += 1
                topics[i] = current_topic

        topic_series = pd.Series([
            "Empty Content" if x == 0 else f"Topic {x}" 
            for x in topics
        ])
        
        return {
            'topics': list(range(1, current_topic + 1)),
            'counts': topic_series.value_counts().to_dict()
        }, topic_series

    def perform_topic_clustering(self, input_data, text_column: str, output_dir: str, threshold: float, batch_size: int = 500, progress_callback=None):
        try:
            # Handle both DataFrame and file path inputs
            df = input_data if isinstance(input_data, pd.DataFrame) else pd.read_excel(input_data)
            
            if progress_callback:
                progress_callback(0.0, "Creating embeddings...")
            
            texts = df[text_column].fillna('').tolist()
            embeddings = self.create_embeddings(
                texts, 
                batch_size,
                lambda p: progress_callback(p * 0.4, "Creating embeddings...") if progress_callback else None
            )
            
            if progress_callback:
                progress_callback(0.4, "Calculating similarities...")
            
            cosine_sim = self.calculate_cosine_similarity(
                embeddings, 
                batch_size=batch_size,
                progress_callback=lambda p: progress_callback(0.4 + p * 0.4, "Calculating similarities...") if progress_callback else None
            )
            
            if progress_callback:
                progress_callback(0.8, "Processing clusters...")
            
            threshold_data, topic_series = self._process_threshold(df, text_column, threshold, cosine_sim)
            
            df['Topics'] = topic_series
            
            if output_dir:
                output_filename = os.path.join(output_dir, "topic_clustering_results.xlsx")
                with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Results', index=False)
            
            if progress_callback:
                progress_callback(1.0, "Complete!")
            
            return df
            
        except Exception as e:
            if progress_callback:
                progress_callback(0, f"Error: {str(e)}")
            return None
