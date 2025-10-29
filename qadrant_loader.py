from chunk import get_chunks
from embeddig import generate_embedding
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
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
QDRANT_URL = os.getenv("QDRANT_URL")

def setup_qdrant_collection_improved():
    """Configura Qdrant con métodos actualizados - VERSIÓN CORREGIDA"""
    
    # === PASO 1: Obtener chunks mejorados ===
    print("📄 Extrayendo chunks mejorados desde el JSON...")
    chunks = get_chunks()
    
    if not chunks:
        return None, None

    # === PASO 2 y 3: Generar embeddings ===
    embeddings, model = generate_embedding(chunks)
    print(f"✅ {len(embeddings)} embeddings generados (dimensión: {len(embeddings[0])})")

    # === PASO 4: Conectar a Qdrant Docker - CAMBIO PRINCIPAL ===
    print("🔗 Conectando a Qdrant Docker...")
    try:
        qdrant = QdrantClient(url=QDRANT_URL)  # ← Conexión al servidor Docker
        
        # Verificar si la colección existe y crearla
        collection_exists = qdrant.collection_exists(collection_name=COLLECTION_NAME)
        if collection_exists:
            print("📂 Colección existente encontrada")
            # Opcional: eliminar si quieres empezar fresco
            # qdrant.delete_collection(collection_name=COLLECTION_NAME)
            # print("♻️ Colección existente eliminada")
        else:
            qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE)
            )
            print("✅ Colección creada en Qdrant Docker")
            
    except Exception as e:
        print(f"❌ Error conectando a Qdrant Docker: {e}")
        print("💡 Asegúrate de que Docker esté ejecutándose con: docker run -p 6333:6333 qdrant/qdrant")
        return None, None

    # === PASO 5: Cargar datos ===
    print("⬆️ Subiendo datos a Qdrant...")
    
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
       
        point = PointStruct(
            id=i,  # ID único
            vector=embedding.tolist(),  # Vector de embeddings
            payload={
                **chunk["metadata"],  # Toda la metadata
                "content": chunk["content"],
                "uploaded_at": datetime.now().isoformat(),
                "chunk_id": i
            }
        )
        points.append(point)
    
    try:
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=points  # Lista de PointStruct
        )
        print(f"✅ {len(points)} puntos subidos correctamente")
    except Exception as e:
        print(f"❌ Error subiendo datos: {e}")
        return None, None

    return qdrant, model

if __name__ == "__main__":
    qadrant,model = setup_qdrant_collection_improved()
    print("qadrant:\n",qadrant)
    print("model:\n",model)