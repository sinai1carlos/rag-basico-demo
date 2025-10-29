import json
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from datetime import datetime
from dotenv import load_dotenv  # Cambio aquí: importar load_dotenv
import os 
# Cargar las variables del archivo .env
load_dotenv()
# === CONFIGURACIÓN ===
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

def generate_embedding(chunks):
    # === Inicializar modelo ===
    print("⚙️ Cargando modelo de embeddings...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # === Generar embeddings ===
    texts = [chunk["content"] for chunk in chunks]
    print("🧠 Generando embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    return embeddings, model
