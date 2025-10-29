from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import json
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from datetime import datetime
from dotenv import load_dotenv
import os
import requests
import time

# Cargar variables de entorno
load_dotenv()

# === CONFIGURACIÓN ===
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documentos_clima")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "gpt-3.5-turbo")

# Inicializar FastAPI
app = FastAPI(
    title="Sistema RAG - Cambio Climático API",
    description="API para consultas inteligentes sobre cambio climático usando RAG",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Modelos Pydantic para request/response
class RAGRequest(BaseModel):
    question: str
    limit: Optional[int] = 5
    use_llm: Optional[bool] = True
    score_threshold: Optional[float] = 0.3

class SearchResult(BaseModel):
    score: float
    content: str
    chapter_title: str
    section_title: str
    metadata: Dict[str, Any]

class RAGResponse(BaseModel):
    question: str
    answer: Optional[str] = None
    search_results: List[SearchResult]
    top_result: SearchResult
    processing_time: float
    total_results: int
    status: str

class HealthResponse(BaseModel):
    status: str
    qdrant_connected: bool
    embedding_model_loaded: bool
    openrouter_available: bool
    collection_info: Optional[Dict[str, Any]] = None
    timestamp: str

# Variables globales para las conexiones
qdrant_client = None
embedding_model = None

def initialize_services():
    """Inicializar conexiones a Qdrant y modelo de embeddings"""
    global qdrant_client, embedding_model
    
    try:
        # Conectar a Qdrant
        print("🔗 Conectando a Qdrant...")
        qdrant_client = QdrantClient(url=QDRANT_URL)
        
        # Verificar conexión
        collections = qdrant_client.get_collections()
        print(f"✅ Qdrant conectado. Colecciones: {[col.name for col in collections.collections]}")
        
        # Cargar modelo de embeddings
        print(f"⚙️ Cargando modelo de embeddings: {EMBEDDING_MODEL}")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print("✅ Modelo de embeddings cargado correctamente")
        
        return True
    except Exception as e:
        print(f"❌ Error inicializando servicios: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Inicializar servicios al iniciar la API"""
    print("🚀 Iniciando API RAG...")
    if not initialize_services():
        print("❌ No se pudieron inicializar los servicios")

def search_documents(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Buscar documentos relevantes en Qdrant"""
    try:
        if not qdrant_client or not embedding_model:
            raise HTTPException(status_code=500, detail="Servicios no inicializados")
        
        # Generar embedding de la consulta
        query_embedding = embedding_model.encode([query]).tolist()[0]
        
        # Buscar en Qdrant
        results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=limit
        ).points
        
        # Formatear resultados
        formatted_results = []
        for result in results:
            formatted_results.append({
                "score": result.score,
                "content": result.payload.get('content', ''),
                "chapter_title": result.payload.get('chapter_title', ''),
                "section_title": result.payload.get('section_title', 'Sección sin título'),
                "metadata": {
                    "section_type": result.payload.get('section_type', ''),
                    "word_count": result.payload.get('word_count', 0),
                    "key_terms": result.payload.get('key_terms', []),
                    "chapter_number": result.payload.get('chapter_number', ''),
                    "document_title": result.payload.get('document_title', '')
                }
            })
        
        return formatted_results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la búsqueda: {str(e)}")

def generate_llm_answer(question: str, context_chunks: List[Dict]) -> Optional[str]:
    """Generar respuesta usando OpenRouter LLM"""
    if not OPENROUTER_API_KEY:
        return None
    
    try:
        # Combinar los chunks en contexto
        context = "\n\n".join([chunk['content'] for chunk in context_chunks])
        
        # Crear el prompt mejorado
        prompt = f"""
        Eres un asistente especializado en cambio climático. Responde la pregunta del usuario basándote ÚNICAMENTE en el contexto proporcionado.

        CONTEXTO:
        {context}

        PREGUNTA: {question}

        INSTRUCCIONES CRÍTICAS:
        - Responde en el MISMO IDIOMA que la pregunta
        - Usa ÚNICAMENTE información del contexto proporcionado
        - Sé conciso y preciso (máximo 200 palabras)
        - Si el contexto no contiene información relevante, di específicamente "No encontré información suficiente en la base de conocimiento para responder esta pregunta"
        - No inventes información ni cites fuentes externas
        - Si el contexto está en inglés pero la pregunta en español, traduce la información al español

        RESPUESTA:
        """
        
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/tu-usuario/rag-sistema",
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.3
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            print(f"❌ Error en OpenRouter: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error llamando al LLM: {e}")
        return None

# Endpoints de la API
@app.get("/", include_in_schema=False)
async def root():
    """Página de inicio de la API"""
    return {
        "message": "🌍 Bienvenido a la API del Sistema RAG - Cambio Climático",
        "version": "1.0.0",
        "description": "Sistema de Retrieval-Augmented Generation para consultas sobre cambio climático",
        "endpoints": {
            "health": "/health - Estado del sistema",
            "query": "/query - Consulta RAG completa",
            "search": "/search - Solo búsqueda semántica",
            "docs": "/docs - Documentación interactiva"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Verificar el estado del sistema"""
    try:
        # Verificar Qdrant
        qdrant_connected = False
        collection_info = None
        
        if qdrant_client:
            try:
                collections = qdrant_client.get_collections()
                qdrant_connected = any(col.name == COLLECTION_NAME for col in collections.collections)
                
                # Obtener información de la colección
                info = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
                collection_info = {
                    "points_count": info.points_count,
                    "vectors_count": info.vectors_count,
                    "status": info.status
                }
            except Exception as e:
                print(f"❌ Error verificando Qdrant: {e}")
        
        # Verificar otros servicios
        embedding_loaded = embedding_model is not None
        openrouter_available = OPENROUTER_API_KEY is not None
        
        # Determinar estado general
        status = "healthy" if all([qdrant_connected, embedding_loaded]) else "degraded"
        if not qdrant_connected or not embedding_loaded:
            status = "unhealthy"
        
        return HealthResponse(
            status=status,
            qdrant_connected=qdrant_connected,
            embedding_model_loaded=embedding_loaded,
            openrouter_available=openrouter_available,
            collection_info=collection_info,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en health check: {str(e)}")

@app.post("/query", response_model=RAGResponse)
async def rag_query(request: RAGRequest):
    """
    Consulta RAG completa: búsqueda semántica + generación de respuesta con LLM
    
    - **question**: Pregunta a consultar
    - **limit**: Número máximo de resultados a recuperar (default: 5)
    - **use_llm**: Si se debe generar respuesta con LLM (default: True)
    - **score_threshold**: Umbral mínimo de similitud (default: 0.3)
    """
    start_time = time.time()
    
    try:
        print(f"🔍 Procesando consulta: '{request.question}'")
        
        # Buscar documentos relevantes
        search_results_data = search_documents(request.question, request.limit)
        
        if not search_results_data:
            raise HTTPException(
                status_code=404, 
                detail="No se encontraron resultados relevantes para la consulta"
            )
        
        # Filtrar por score threshold si se especifica
        if request.score_threshold > 0:
            search_results_data = [
                result for result in search_results_data 
                if result['score'] >= request.score_threshold
            ]
            
        if not search_results_data:
            raise HTTPException(
                status_code=404, 
                detail=f"No se encontraron resultados con score >= {request.score_threshold}"
            )
        
        # Generar respuesta con LLM si está activado
        answer = None
        if request.use_llm and OPENROUTER_API_KEY:
            print("🤖 Generando respuesta con LLM...")
            answer = generate_llm_answer(request.question, search_results_data)
        
        # Calcular tiempo de procesamiento
        processing_time = time.time() - start_time
        
        # Convertir a modelos Pydantic
        search_results = [SearchResult(**result) for result in search_results_data]
        
        response = RAGResponse(
            question=request.question,
            answer=answer,
            search_results=search_results,
            top_result=search_results[0],
            processing_time=round(processing_time, 2),
            total_results=len(search_results),
            status="success"
        )
        
        print(f"✅ Consulta procesada en {processing_time:.2f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"❌ Error procesando consulta: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error interno procesando la consulta: {str(e)}"
        )

@app.post("/search", response_model=List[SearchResult])
async def search_only(
    question: str = Query(..., description="Pregunta a buscar"),
    limit: int = Query(5, description="Número máximo de resultados", ge=1, le=20),
    score_threshold: float = Query(0.0, description="Umbral mínimo de similitud", ge=0.0, le=1.0)
):
    """
    Solo búsqueda semántica sin generación de respuesta LLM
    
    - **question**: Pregunta a buscar
    - **limit**: Número máximo de resultados (1-20)
    - **score_threshold**: Umbral mínimo de similitud (0.0-1.0)
    """
    start_time = time.time()
    
    try:
        print(f"🔍 Búsqueda semántica: '{question}'")
        
        results = search_documents(question, limit)
        
        if not results:
            raise HTTPException(
                status_code=404, 
                detail="No se encontraron resultados relevantes"
            )
        
        # Filtrar por score threshold
        if score_threshold > 0:
            results = [result for result in results if result['score'] >= score_threshold]
            
        if not results:
            raise HTTPException(
                status_code=404, 
                detail=f"No se encontraron resultados con score >= {score_threshold}"
            )
        
        # Convertir a modelo Pydantic
        search_results = [SearchResult(**result) for result in results]
        
        processing_time = time.time() - start_time
        print(f"✅ Búsqueda completada en {processing_time:.2f}s - {len(results)} resultados")
        
        return search_results
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error en búsqueda: {e}")
        raise HTTPException(status_code=500, detail=f"Error en búsqueda: {str(e)}")

@app.get("/collection/info")
async def collection_info():
    """Obtener información detallada de la colección en Qdrant"""
    try:
        if not qdrant_client:
            raise HTTPException(status_code=500, detail="Qdrant no conectado")
        
        info = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
        
        return {
            "collection_name": COLLECTION_NAME,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "status": info.status,
            "vectors_config": str(info.vectors_config),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo información: {str(e)}")

@app.get("/config")
async def get_config():
    """Obtener configuración actual del sistema"""
    return {
        "qdrant_url": QDRANT_URL,
        "collection_name": COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "openrouter_available": OPENROUTER_API_KEY is not None,
        "openrouter_model": OPENROUTER_MODEL if OPENROUTER_API_KEY else "No configurado"
    }

if __name__ == "__main__":
    # Inicializar servicios primero
    if initialize_services():
        # Ejecutar servidor
        uvicorn.run(
            "api_rag_fastapi:app",
            host="0.0.0.0",  # Accesible desde cualquier IP
            port=8000,
            reload=True,  # Recarga automática en desarrollo
            log_level="info"
        )
    else:
        print("❌ No se pudo inicializar la API debido a errores en los servicios")