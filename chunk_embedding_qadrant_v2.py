import json
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from datetime import datetime

# === CONFIGURACIÓN ===
JSON_DATA_PATH = "climate_change_structured.json"
COLLECTION_NAME = "documentos_clima"
# EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
QDRANT_URL = "http://localhost:6333"  # ← Nueva configuración

def load_and_improve_json_data(json_path):
    """Carga y mejora los datos del JSON"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ JSON cargado correctamente desde {json_path}")
        
        # Mejorar los datos: agregar títulos donde falten
        for chapter in data['document']['chapters']:
            for section in chapter['sections']:
                if not section.get('section_title') and section['section_type'] != 'main_content':
                    # Crear título basado en el contenido
                    first_words = " ".join(section['content'].split()[:4])
                    section['section_title'] = f"{first_words}..."
        
        return data
    except Exception as e:
        print(f"❌ Error cargando JSON: {e}")
        return None

def extract_chunks_from_json(structured_data):
    """Extrae chunks mejorados del JSON"""
    if not structured_data:
        return []
    
    chunks = []
    
    for chapter in structured_data['document']['chapters']:
        for section in chapter['sections']:
            # Mejorar el título de sección
            section_title = section.get('section_title', '')
            if not section_title and section['section_type'] == 'main_content':
                section_title = f"Introducción - {chapter['chapter_title']}"
            
            chunk_data = {
                'content': section['content'],
                'metadata': {
                    'chapter_number': chapter['chapter_number'],
                    'chapter_title': chapter['chapter_title'],
                    'section_type': section['section_type'],
                    'section_title': section_title,
                    'word_count': section['metadata']['word_count'],
                    'key_terms': section['metadata']['key_terms'],
                    'document_title': structured_data['document']['title'],
                    'content_preview': section['content'][:100] + "..." if len(section['content']) > 100 else section['content']
                }
            }
            chunks.append(chunk_data)
    
    return chunks

def get_chunks():
    """Función principal para obtener chunks mejorados"""
    climate_data = load_and_improve_json_data(JSON_DATA_PATH)
    
    if not climate_data:
        print("❌ No se pudieron cargar los datos.")
        return []
    
    chunks = extract_chunks_from_json(climate_data)
    print(f"✅ {len(chunks)} chunks extraídos y mejorados del JSON")
    
    # Mostrar estadísticas
    chunks_with_titles = sum(1 for chunk in chunks if chunk['metadata']['section_title'])
    print(f"📊 Chunks con títulos: {chunks_with_titles}/{len(chunks)}")
    
    return chunks

def setup_qdrant_collection_improved():
    """Configura Qdrant con métodos actualizados - VERSIÓN CORREGIDA"""
    
    # === PASO 1: Obtener chunks mejorados ===
    print("📄 Extrayendo chunks mejorados desde el JSON...")
    chunks = get_chunks()
    
    if not chunks:
        return None, None

    # === PASO 2: Inicializar modelo ===
    print("⚙️ Cargando modelo de embeddings...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # === PASO 3: Generar embeddings ===
    texts = [chunk["content"] for chunk in chunks]
    print("🧠 Generando embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)
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

    # === PASO 5: Cargar datos - CORRECCIÓN AQUÍ ===
    print("⬆️ Subiendo datos a Qdrant...")
    
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        # CORRECCIÓN: Usar PointStruct en lugar de diccionario
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

def search_improved(qdrant, model, query, collection_name=COLLECTION_NAME, limit=3):
    """Búsqueda con métodos actualizados"""
    query_embedding = model.encode([query]).tolist()[0]
    
    results = qdrant.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=limit
    ).points
    
    return results

def test_search_improved():
    """Prueba mejorada del sistema"""
    print("🚀 Iniciando prueba del sistema RAG...")
    qdrant, model = setup_qdrant_collection_improved()
    
    if not qdrant or not model:
        print("❌ No se pudo inicializar el sistema.")
        return
    
    # Consultas de prueba mejoradas
    test_queries = [
        "¿Qué es el cambio climático?",
        "Gases de efecto invernadero y calentamiento global",
        "Causas principales del cambio climático",
        "Impacto de la deforestación",
        "Agricultura y emisiones de metano"
    ]
    
    print("\n🔍 TEST MEJORADO - Resultados de búsqueda:")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n🎯 CONSULTA: '{query}'")
        
        results = search_improved(qdrant, model, query, limit=3)
        
        if not results:
            print("   ❌ No se encontraron resultados")
            continue
            
        print(f"   ✅ Top {len(results)} resultados:")
        for i, result in enumerate(results):
            payload = result.payload
            print(f"   {i+1}. 📍 {payload.get('section_title', 'Sin título')}")
            print(f"      📖 Capítulo {payload.get('chapter_number')}: {payload.get('chapter_title')}")
            print(f"      🔢 Similitud: {result.score:.3f}")
            print(f"      📝 {payload.get('content_preview', '')}")
            print(f"      📋 Términos clave: {', '.join(payload.get('key_terms', []))}")
            print()

def interactive_search():
    """Modo interactivo para probar búsquedas"""
    print("🔄 Inicializando sistema para búsqueda interactiva...")
    qdrant, model = setup_qdrant_collection_improved()
    
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
    test_search_improved()
    
    # Preguntar si quiere modo interactivo
    response = input("\n¿Quieres probar el modo interactivo? (s/n): ").strip().lower()
    if response in ['s', 'si', 'sí', 'y', 'yes']:
        interactive_search()