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
JSON_DATA_PATH = os.getenv("JSON_DATA_PATH")

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

