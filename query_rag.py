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
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

def connect_to_existing_qdrant():
    """Conecta a Qdrant ya configurado"""

    # Conectar a Qdrant
    qdrant = QdrantClient(url=QDRANT_URL)
    
    # Cargar modelo de embeddings
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Verificar que la colección existe
    if not qdrant.collection_exists(collection_name="documentos_clima"):
        print("❌ La colección no existe. Ejecuta setup_qdrant_collection_improved() primero.")
        return None, None
    
    return qdrant, model

def search_improved(qdrant, model, query, collection_name=COLLECTION_NAME, limit=3):
    """Búsqueda con métodos actualizados"""
    query_embedding = model.encode([query]).tolist()[0]
    
    results = qdrant.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=limit
    ).points
    
    return results

def interactive_search():
    """Modo interactivo para probar búsquedas"""
    print("🔄 Inicializando sistema para búsqueda interactiva...")
    qdrant, model = connect_to_existing_qdrant()
    
    if not qdrant or not model:
        print("❌ No se pudo inicializar el sistema para búsqueda interactiva.")
        return
    
    print("\n" + "=" * 60)
    print("🔍 MODO INTERACTIVO - Sistema RAG de Cambio Climático")
    print("=" * 60)
    print("Escribe tus preguntas sobre cambio climático")
    print("Comandos: 'salir' para terminar, 'estado' para ver estadísticas")
    
    while True:
        try:
            query = input("\n🎯 Tu pregunta: ").strip()
            
            if query.lower() in ['salir', 'exit', 'quit']:
                print("👋 ¡Hasta pronto!")
                break
                
            if query.lower() == 'estado':
                # Mostrar estadísticas de la colección
                try:
                    info = qdrant.get_collection(collection_name=COLLECTION_NAME)
                    print(f"📊 Colección: {COLLECTION_NAME}")
                    print(f"   Puntos almacenados: {info.points_count}")
                    print(f"   Vectores: {info.vectors_count}")
                except Exception as e:
                    print(f"❌ Error obteniendo estadísticas: {e}")
                continue
                
            if not query:
                continue
                
            # Buscar
            print("🔍 Buscando información relevante...")
            results = search_improved(qdrant, model, query, limit=5)
            
            if not results:
                print("   ❌ No encontré información relevante para tu pregunta.")
                continue
                
            print(f"\n   ✅ Encontré {len(results)} resultados relevantes:")
            
            # Mostrar el mejor resultado de manera destacada
            best_result = results[0]
            best_payload = best_result.payload
            
            print(f"\n   🏅 MEJOR RESULTADO (similitud: {best_result.score:.3f}):")
            print(f"   📖 {best_payload.get('chapter_title')}")
            print(f"   📑 {best_payload.get('section_title', 'Sección sin título')}")
            print(f"   📝 {best_payload['content']}")
            
            # Mostrar otros resultados si existen
            if len(results) > 1:
                print(f"\n   📚 OTROS RESULTADOS RELEVANTES:")
                for i, result in enumerate(results[1:], 2):
                    payload = result.payload
                    print(f"   {i}. {payload.get('section_title', 'Sin título')} (sim: {result.score:.3f})")
                    
        except KeyboardInterrupt:
            print("\n👋 ¡Hasta pronto!")
            break
        except Exception as e:
            print(f"❌ Error durante la búsqueda: {e}")

if __name__ == "__main__":
    # Ejecutar prueba automática
    # test_search_improved()
    interactive_search()
    # Preguntar si quiere modo interactivo
    # response = input("\n¿Quieres probar el modo interactivo? (s/n): ").strip().lower()
    # if response in ['s', 'si', 'sí', 'y', 'yes']:
        # interactive_search()